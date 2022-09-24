import functools
import time
import os

from einops import repeat

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from typing import Any, Optional, Callable
from tqdm import tqdm, trange

import flax
from flax.training import train_state
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import checkpoints
from flax import jax_utils

import optax

import jax.numpy as jnp
import numpy as np
import jax 

import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds

import wandb

import unet
import utils
from sampling import sample_loop, ddpm_sample_step, model_predict


def flatten(x):
  return x.reshape(x.shape[0], -1)

def l2_loss(logit, target):
    return (logit - target)**2

def l1_loss(logit, target): 
    return jnp.abs(logit - target)

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def crop_resize(image, resolution):
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
      image,
      size=(resolution, resolution),
      antialias=True,
      method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def get_dataset(rng, config):
    
    if config.data.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    
    batch_size = config.data.batch_size //jax.process_count()

    platform = jax.local_devices()[0].platform
    if config.training.half_precision:
        if platform == 'tpu':
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else: input_dtype = tf.float32

    dataset_builder = tfds.builder(config.data.dataset)
    dataset_builder.download_and_prepare()

    def preprocess_fn(d):
        img = d['image']
        img = crop_resize(img, config.data.image_size)
        img = tf.image.flip_left_right(img)
        img= tf.image.convert_image_dtype(img, input_dtype)
        return({'image':img})
    
    # create split for current process 
    train_examples = dataset_builder.info.splits['train'].num_examples
    split_size = train_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = f'train[{start}:{start + split_size}]'

    ds = dataset_builder.as_dataset(split=split)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    ds.with_options(options)

    if config.data.cache:
        ds= ds.cache()   

    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size , seed=0)

    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # (local_devices * device_batch_size), height, width, c
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(10)

    def scale_and_reshape(xs):
        local_device_count = jax.local_device_count()
        def _scale_and_reshape(x):
           # Use _numpy() for zero-copy conversion between TF and NumPy.
           x = x._numpy()  # pylint: disable=protected-access
           # normalize to [-1,1]
           x = normalize_to_neg_one_to_one(x)
          # reshape (batch_size, height, width, channels) to
         # (local_devices, device_batch_size, height, width, 3)
           return x.reshape((local_device_count, -1) + x.shape[1:])

        return jax.tree_map(_scale_and_reshape, xs)

    it = map(scale_and_reshape, ds)
    it = jax_utils.prefetch_to_device(it, 2)

    return(it)   


def create_model(*, model_cls, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(dtype=model_dtype, **kwargs)


def initialized(key, image_size,image_channel, model):

  input_shape = (1, image_size, image_size, image_channel)

  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init(
      {'params': key}, 
      jnp.ones(input_shape, model.dtype), # x noisy image
      jnp.ones(input_shape[:1], model.dtype) # t
      )

  return variables['params']


class TrainState(train_state.TrainState):
  params_ema: Any = None
  dynamic_scale: Optional[dynamic_scale_lib.DynamicScale] = None


def create_train_state(rng, config: ml_collections.ConfigDict):
  """Creates initial `TrainState`."""

  dynamic_scale = None
  platform = jax.local_devices()[0].platform

  if config.training.half_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  else:
    dynamic_scale = None

  model = create_model(
      model_cls=unet.Unet, 
      half_precision=config.training.half_precision,
      dim = config.model.dim, 
      out_dim =  config.data.channels,
      dim_mults = config.model.dim_mults)

  rng, rng_params = jax.random.split(rng)
  image_size = config.data.image_size
  input_dim = config.data.channels * 2 if config.ddpm.self_condition else config.data.channels
  params = initialized(rng_params, image_size, input_dim, model)

  tx = create_optimizer(config.optim)

  state = TrainState.create(
      apply_fn=model.apply, 
      params=params, 
      tx=tx, 
      params_ema=params,
      dynamic_scale=dynamic_scale)

  return state


def create_optimizer(config):

    if config.optimizer == 'Adam':
        optimizer = optax.adam(
            learning_rate = config.lr , b1=config.beta1, b2 = config.beta2, 
            eps=config.eps)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def get_loss_fn(config):

    if config.training.loss_type == 'l1' :
        loss_fn = l1_loss
    elif config.training.loss_type == 'l2':
        loss_fn = l2_loss
    else:
        raise NotImplementedError(
           f'loss_type {config.training.loss_tyoe} not supported yet!')

    return loss_fn


def create_ema_decay_schedule(config):

    def ema_decay_schedule(step):
        count = jnp.clip(step - config.update_after_step - 1, a_min = 0.)
        value = 1 - (1 + count / config.inv_gamma) ** - config.power 
        ema_rate = jnp.clip(value, a_min = config.min_value, a_max = config.beta)
        return ema_rate

    return ema_decay_schedule


def q_sample(x, t, noise, ddpm_params):

    sqrt_alpha_bar = ddpm_params['sqrt_alphas_bar'][t, None, None, None]
    sqrt_1m_alpha_bar = ddpm_params['sqrt_1m_alphas_bar'][t,None,None,None]
    x_t = sqrt_alpha_bar * x + sqrt_1m_alpha_bar * noise

    return x_t


# train step
def p_loss(rng, state, batch, ddpm_params, loss_fn, self_condition=False, is_pred_x0=False, pmap_axis='batch'):
    
    # run the forward diffusion process to generate noisy image x_t at timestep t
    x = batch['image']
    assert x.dtype in [jnp.float32, jnp.float64]
    
    # create batched timesteps: t with shape (B,)
    B, H, W, C = x.shape
    rng, t_rng = jax.random.split(rng)
    batched_t = jax.random.randint(t_rng, shape=(B,), dtype = jnp.int32, minval=0, maxval= len(ddpm_params['betas']))
   
    # sample a noise (input for q_sample)
    rng, noise_rng = jax.random.split(rng)
    noise = jax.random.normal(noise_rng, x.shape)
    # if is_pred_x0 == True, the target for loss calculation is x, else noise
    target = x if is_pred_x0 else noise

    # generate the noisy image (input for denoise model)
    x_t = q_sample(x, batched_t, noise, ddpm_params)
    
    # if doing self-conditioning, 50% of the time first estimate x_0 = f(x_t, 0, t) and then use the estimated x_0 for Self-Conditioning
    # we don't backpropagate through the estimated x_0 (exclude from the loss calculation)
    # this technique will slow down training by 25%, but seems to lower FID significantly  
    if self_condition:

        rng, condition_rng = jax.random.split(rng)
        zeros = jnp.zeros_like(x_t)

        # self-conditioning 
        def estimate_x0(_):
            x0, _ = model_predict(state, x_t, zeros, batched_t, ddpm_params, self_condition, is_pred_x0, use_ema=False)
            return x0

        x0 = jax.lax.cond(
            jax.random.uniform(condition_rng, shape=(1,))[0] < 0.5,
            estimate_x0,
            lambda _ :zeros,
            None)
                
        x_t = jnp.concatenate([x_t, x0], axis=-1)
    
    p2_loss_weight = ddpm_params['p2_loss_weight']

    def compute_loss(params):
        pred = state.apply_fn({'params':params}, x_t, batched_t)
        loss = loss_fn(flatten(pred),flatten(target))
        loss = jnp.mean(loss, axis= 1)
        assert loss.shape == (B,)
        loss = loss * p2_loss_weight[batched_t]
        return loss.mean()
    
    dynamic_scale = state.dynamic_scale

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(compute_loss, axis_name=pmap_axis)
        dynamic_scale, is_fin, loss, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(compute_loss)
        loss, grads = grad_fn(state.params)
        #  Re-use same axis_name as in the call to `pmap(...train_step,axis=...)` in the train function
        grads = jax.lax.pmean(grads, axis_name=pmap_axis)
    
    loss = jax.lax.pmean(loss, axis_name=pmap_axis)
    loss_ema = jax.lax.pmean(compute_loss(state.params_ema), axis_name=pmap_axis)

    metrics = {'loss': loss,
               'loss_ema': loss_ema}

    new_state = state.apply_gradients(grads=grads)

    if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_map(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state),
            params=jax.tree_map(
                functools.partial(jnp.where, is_fin),
                new_state.params,
                state.params),
            dynamic_scale=dynamic_scale)
        metrics['scale'] = dynamic_scale.scale
    
     
    return new_state, metrics



def copy_params_to_ema(state):
   state = state.replace(params_ema = state.params)
   return state

def apply_ema_decay(state, ema_decay):
    params_ema = jax.tree_map(lambda p_ema, p: p_ema * ema_decay + p * (1. - ema_decay), state.params_ema, state.params)
    state = state.replace(params_ema = params_ema)
    return state


def load_wandb_model(state, workdir, wandb_artifact):
    artifact = wandb.run.use_artifact(wandb_artifact, type='ddpm_model')
    artifact_dir = artifact.download(workdir)
    return checkpoints.restore_checkpoint(artifact_dir, state)


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


def train(config: ml_collections.ConfigDict, 
          workdir: str,
          wandb_artifact: str = None) -> TrainState:
  """Execute model training loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
    # create writer 
  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)
    # set up wandb run
  if config.wandb.log_train and jax.process_index() == 0:
      wandb_config = utils.to_wandb_config(config)
      wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            job_type=config.wandb.job_type,
            config=wandb_config)
      # set default x-axis as 'train/step'
      #wandb.define_metric("*", step_metric="train/step")

  sample_dir = os.path.join(workdir, "samples")

  rng = jax.random.PRNGKey(config.seed)

  rng, d_rng = jax.random.split(rng) 
  train_iter = get_dataset(d_rng, config)
  
  num_steps = config.training.num_train_steps
  
  rng, state_rng = jax.random.split(rng)
  state = create_train_state(state_rng, config)
  if wandb_artifact is not None:
      logging.info(f'loading model from wandb: {wandb_artifact}')
      state = load_wandb_model(state, workdir, wandb_artifact)
  else:
      state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  state = jax_utils.replicate(state)
  
  loss_fn = get_loss_fn(config)
 
  ddpm_params = utils.get_ddpm_params(config.ddpm)
  ema_decay_fn = create_ema_decay_schedule(config.ema)
  train_step = functools.partial(p_loss, ddpm_params=ddpm_params, loss_fn =loss_fn, self_condition=config.ddpm.self_condition, is_pred_x0=config.ddpm.pred_x0, pmap_axis ='batch')
  p_train_step = jax.pmap(train_step, axis_name = 'batch')
  p_apply_ema = jax.pmap(apply_ema_decay, in_axes=(0, None), axis_name = 'batch')
  p_copy_params_to_ema = jax.pmap(copy_params_to_ema, axis_name='batch')

  train_metrics = []
  hooks = []

  sample_step = functools.partial(ddpm_sample_step, ddpm_params=ddpm_params, self_condition=config.ddpm.self_condition, is_pred_x0=config.ddpm.pred_x0)
  p_sample_step = jax.pmap(sample_step, axis_name='batch')

  if jax.process_index() == 0:
      hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for step, batch in zip(tqdm(range(step_offset, num_steps)), train_iter):
      rng, *train_step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      train_step_rng = jnp.asarray(train_step_rng)
      state, metrics = p_train_step(train_step_rng, state, batch)
      for h in hooks:
          h(step)
      if step == step_offset:
          logging.info('Initial compilation completed.')
          logging.info(f"Number of devices: {batch['image'].shape[0]}")
          logging.info(f"Batch size per device {batch['image'].shape[1]}")
          logging.info(f"input shape: {batch['image'].shape[2:]}")

      # update state.params_ema
      if (step + 1) <= config.ema.update_after_step:
          state = p_copy_params_to_ema(state)
      elif (step + 1) % config.ema.update_every == 0:
          ema_decay = ema_decay_fn(step)
          logging.info(f'update ema parameters with decay rate {ema_decay}')
          state =  p_apply_ema(state, ema_decay)

      if config.training.get('log_every_steps'):
          train_metrics.append(metrics)
          if (step + 1) % config.training.log_every_steps == 0:
              train_metrics = common_utils.get_metrics(train_metrics)
              summary = {
                    f'train/{k}': v
                    for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
                }
              summary['time/seconds_per_step'] =  (time.time() - train_metrics_last_t) /config.training.log_every_steps

              writer.write_scalars(step + 1, summary)
              train_metrics = []
              train_metrics_last_t = time.time()

              if config.wandb.log_train:
                  wandb.log({
                      "train/step": step, ** summary
                  })
      
      # Save a checkpoint periodically and generate samples.
      if (step + 1) % config.training.save_and_sample_every == 0 or step + 1 == num_steps:
          # generate and save sampling 
          logging.info(f'generating samples....')
          samples = []
          for i in trange(0, config.training.num_sample, config.data.batch_size):
              rng, sample_rng = jax.random.split(rng)
              samples.append(sample_loop(sample_rng, state, tuple(batch['image'].shape), p_sample_step, config.ddpm.timesteps))
          samples = jnp.concatenate(samples) # num_devices, batch, H, W, C
          
          this_sample_dir = os.path.join(sample_dir, f"iter_{step}_host_{jax.process_index()}")
          tf.io.gfile.makedirs(this_sample_dir)
          
          with tf.io.gfile.GFile(
              os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
            samples_array = utils.save_image(samples, config.training.num_sample, fout, padding=2)
            if config.wandb.log_sample:
                utils.wandb_log_image(samples_array, step+1)
          # save the chceckpoint
          save_checkpoint(state, workdir)
          if step + 1 == num_steps and config.wandb.log_model:
              utils.wandb_log_model(workdir, step+1)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return(state)







  

