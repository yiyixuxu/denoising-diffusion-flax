import jax.numpy as jnp
import jax 
from flax import jax_utils

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def get_posterior_mean_variance(img, t, pred_noise, ddpm_params):

    betas = ddpm_params['betas']
    alphas = ddpm_params['alphas']
    alphas_bar = ddpm_params['alphas_bar']
    sqrt_alphas_bar = ddpm_params['sqrt_alphas_bar']

    # calculate x_0 (needed for self-condition calculation)
    x0 = 1. / sqrt_alphas_bar[t] * img -  jnp.sqrt(1./alphas_bar[t]-1) * pred_noise
    x0 = jnp.clip(x0, -1., 1.)
    
    # only needed when t > 0
    coef_x0 = betas[t] * sqrt_alphas_bar[t-1] / (1. - alphas_bar[t])
    coef_xt = (1. - alphas_bar[t-1]) * jnp.sqrt(alphas[t]) / ( 1- alphas_bar[t])        
    posterior_mean = coef_x0 * x0 + coef_xt * img
        
    posterior_variance = betas[t] * (1 - alphas_bar[t-1]) / (1. - alphas_bar[t])
    posterior_log_variance = jnp.log(jnp.clip(posterior_variance, a_min = 1e-20))

    return x0, posterior_mean, posterior_log_variance


# eval step
def model_predict(state, x, t):
    variables = {'params': state.params_ema}
    logits = state.apply_fn(
        variables, x, t)
    return logits

def ddpm_sample_step(state, rng, x, t, x0, ddpm_params, self_condition=False):
    # predicted_noise
 
    batched_t = jnp.ones((x.shape[0],), dtype=jnp.int32) * t
    
    if self_condition:
        v = model_predict(state, jnp.concatenate([x,x0], axis=-1), batched_t) 
    else:
        v = model_predict(state, x, batched_t)
       
    x0, posterior_mean, posterior_log_variance = get_posterior_mean_variance(x, t, v, ddpm_params)
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

