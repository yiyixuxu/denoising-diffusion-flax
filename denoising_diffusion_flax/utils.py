
import jax.numpy as jnp
import numpy as np
import jax 
import math
from PIL import Image
import wandb
from ml_collections import ConfigDict


def cosine_beta_schedule(timesteps):
    """Return cosine schedule 
    as proposed in https://arxiv.org/abs/2102.09672 """
    s=0.008
    max_beta=0.999
    ts = jnp.linspace(0, 1, timesteps + 1)
    alphas_bar = jnp.cos((ts + s) / (1 + s) * jnp.pi /2) ** 2
    alphas_bar = alphas_bar/alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return(jnp.clip(betas, 0, max_beta))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = jnp.linspace(
        beta_start, beta_end, timesteps, dtype=jnp.float64)
    return(betas)

def get_ddpm_params(config):
    schedule_name = config.beta_schedule
    timesteps = config.timesteps
    p2_loss_weight_gamma = config.p2_loss_weight_gamma
    p2_loss_weight_k = config.p2_loss_weight_gamma

    if schedule_name == 'linear':
        betas = linear_beta_schedule(timesteps)
    elif schedule_name == 'cosine':
        betas = cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f'unknown beta schedule {schedule_name}')
    assert betas.shape == (timesteps,)
    alphas = 1. - betas
    alphas_bar = jnp.cumprod(alphas, axis=0)
    sqrt_alphas_bar = jnp.sqrt(alphas_bar)
    sqrt_1m_alphas_bar= jnp.sqrt(1. - alphas_bar)
    
    # calculate p2 reweighting
    p2_loss_weight=  (p2_loss_weight_k + alphas_bar / (1 - alphas_bar)) ** -p2_loss_weight_gamma


    return {
      'betas': betas,
      'alphas': alphas,
      'alphas_bar': alphas_bar,
      'sqrt_alphas_bar': sqrt_alphas_bar,
      'sqrt_1m_alphas_bar': sqrt_1m_alphas_bar,
      'p2_loss_weight': p2_loss_weight
  }



def make_grid(samples, n_samples, padding=2, pad_value=0.0):

  ndarray = samples.reshape((-1, *samples.shape[2:]))[:n_samples]
  nrow = int(np.sqrt(ndarray.shape[0]))

  if not (isinstance(ndarray, jnp.ndarray) or
          (isinstance(ndarray, list) and
           all(isinstance(t, jnp.ndarray) for t in ndarray))):
    raise TypeError("array_like of tensors expected, got {}".format(
        type(ndarray)))

  ndarray = jnp.asarray(ndarray)

  if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
    ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

  # make the mini-batch of images into a grid
  nmaps = ndarray.shape[0]
  xmaps = min(nrow, nmaps)
  ymaps = int(math.ceil(float(nmaps) / xmaps))
  height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] +
                                                       padding)
  num_channels = ndarray.shape[3]
  grid = jnp.full(
      (height * ymaps + padding, width * xmaps + padding, num_channels),
      pad_value).astype(jnp.float32)
  k = 0
  for y in range(ymaps):
    for x in range(xmaps):
      if k >= nmaps:
        break
      grid = grid.at[y * height + padding:(y + 1) * height,
                     x * width + padding:(x + 1) * width].set(ndarray[k])
      k = k + 1
  return grid


def save_image(samples, n_samples, fp, padding=2, pad_value=0.0, format=None):
  """Make a grid of images and Save it into an image file.

  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C).
    fp: A filename(string) or file object.
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format(Optional):  If omitted, the format to use is determined from the
      filename extension. If a file object was used instead of a filename, this
      parameter should always be used.
  """

  grid = make_grid(samples, n_samples, padding, pad_value)
  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
  ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
  ndarr = np.array(ndarr)
  im = Image.fromarray(ndarr)
  im.save(fp, format=format)

  return ndarr

def wandb_log_image(samples_array, step):
    sample_images = wandb.Image(samples_array, caption = f"step {step}")
    wandb.log({'samples':sample_images })

def wandb_log_model(workdir, step):
    artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="ddpm_model")
    artifact.add_file( f"{workdir}/checkpoint_{step}")
    wandb.run.log_artifact(artifact)


def to_wandb_config(d: ConfigDict, parent_key: str = '', sep: str ='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, ConfigDict):
            items.extend(to_wandb_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)