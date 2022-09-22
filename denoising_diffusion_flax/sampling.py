import jax.numpy as jnp
import jax 
from flax import jax_utils


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def noise_to_x0(noise, xt, batched_t, ddpm):
    assert batched_t.shape[0] == xt.shape[0] == noise.shape[0] # make sure all has batch dimension
    sqrt_alpha_bar = ddpm['sqrt_alphas_bar'][batched_t, None, None, None]
    alpha_bar= ddpm['alphas_bar'][batched_t, None, None, None]
    x0 = 1. / sqrt_alpha_bar * xt -  jnp.sqrt(1./alpha_bar-1) * noise
    return x0


def x0_to_noise(x0, xt, batched_t, ddpm):
    assert batched_t.shape[0] == xt.shape[0] == x0.shape[0] # make sure all has batch dimension
    sqrt_alpha_bar = ddpm['sqrt_alphas_bar'][batched_t, None, None, None]
    alpha_bar= ddpm['alphas_bar'][batched_t, None, None, None]
    noise = (1. / sqrt_alpha_bar * xt - x0) /jnp.sqrt(1./alpha_bar-1)
    return noise


def get_posterior_mean_variance(img, t, x0, v, ddpm_params):

    beta = ddpm_params['betas'][t, None,None,None]
    alpha = ddpm_params['alphas'][t, None,None,None]
    alpha_bar = ddpm_params['alphas_bar'][t, None,None,None]
    alpha_bar_last = ddpm_params['alphas_bar'][t-1, None,None,None]
    sqrt_alpha_bar_last = ddpm_params['sqrt_alphas_bar'][t-1, None,None,None]

    # only needed when t > 0
    coef_x0 = beta * sqrt_alpha_bar_last / (1. - alpha_bar)
    coef_xt = (1. - alpha_bar_last) * jnp.sqrt(alpha) / ( 1- alpha_bar)        
    posterior_mean = coef_x0 * x0 + coef_xt * img
        
    posterior_variance = beta * (1 - alpha_bar_last) / (1. - alpha_bar)
    posterior_log_variance = jnp.log(jnp.clip(posterior_variance, a_min = 1e-20))

    return posterior_mean, posterior_log_variance


# called by p_loss and ddpm_sample_step - both use pmap
def model_predict(state, x, x0, t, ddpm_params, self_condition, is_pred_x0, use_ema=True):
    if use_ema:
        variables = {'params': state.params_ema}
    else:
        variables = {'params': state.params}
    
    if self_condition:
        pred = state.apply_fn(variables, jnp.concatenate([x, x0],axis=-1), t)
    else:
        pred = state.apply_fn(variables, x, t)

    if is_pred_x0: # if the objective is is_pred_x0, pred == x0_pred
        x0_pred = pred
        noise_pred =  x0_to_noise(pred, x, t, ddpm_params)
    else:
        noise_pred = pred
        x0_pred = noise_to_x0(pred, x, t, ddpm_params)
    
    return x0_pred, noise_pred


def ddpm_sample_step(state, rng, x, t, x0_last, ddpm_params, self_condition=False, is_pred_x0=False):
 
    batched_t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t
    
    if self_condition:
        x0, v = model_predict(state, x, x0_last, batched_t, ddpm_params, self_condition, is_pred_x0, use_ema=True) 
    else:
        x0, v = model_predict(state, x, None, batched_t,ddpm_params, self_condition, is_pred_x0, use_ema=True)
    
    # make sure x0 between [-1,1]
    x0 = jnp.clip(x0, -1., 1.)

    posterior_mean, posterior_log_variance = get_posterior_mean_variance(x, t, x0, v, ddpm_params)
    x = posterior_mean + jnp.exp(0.5 *  posterior_log_variance) * jax.random.normal(rng, x.shape) 

    return x, x0


def sample_loop(rng, state, shape, p_sample_step, timesteps):
    
    # shape include the device dimension: (device, per_device_batch_size, H,W,C)
    rng, x_rng = jax.random.split(rng)
    list_x0 = []
    # generate the initial sample (pure noise)
    x = jax.random.normal(x_rng, shape)
    x0 = jnp.zeros_like(x) # initialize x0 for self-conditioning
    # sample step
    for t in reversed(jnp.arange(timesteps)):
        rng, *step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        step_rng = jnp.asarray(step_rng)
        x, x0 = p_sample_step(state, step_rng, x, jax_utils.replicate(t), x0)
        list_x0.append(x0)
    # normalize to [0,1]
    img = unnormalize_to_zero_to_one(jnp.asarray(x0))

    return img

