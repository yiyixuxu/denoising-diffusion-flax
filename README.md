# denoising-diffusion-flax

Denoising Diffusion Probabilistic Model in Flax 

This implementation is based on [lucidrains](https://github.com/lucidrains)'s [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch), where he implemented the original DDPM model proposed from paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), as well as latest research findings


### Running locally

```shell
python main.py --workdir=./fashion_mnist_cpu --mode=train --config=configs/fashion_mnist_cpu.py 
```

#### Overriding parameters on the command line

Specify a hyperparameter configuration by the means of setting `--config` flag.
Configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
python main.py --workdir=./fashion_mnist_cpu --config=configs/fashion_mnist_cpu.py  \
--config.training.num_train_steps=100
```

#### load a pre-trained model from W&B artifact

```
python main.py --workdir=./fashion_mnist_wandb --mode=train --wandb_artifact=yiyixu/ddpm-flax-fashion-mnist/model-3j8xvqwf:v0 --config=configs/fashion_mnist_cpu.py 
```

### Google Cloud TPU

If you're new to Jax/Flax ecosystem, you can apply for TPU free trial here https://sites.research.google/trc/about/

See below for commands to set up a single VM with 8 TPUs attached
(`--accelerator-type v3-8`). For more details about how to set up and
use TPUs, refer to Cloud docs for
[single VM setup](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)(https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm).


First create a single TPUv3-8 VM and connect to it:

(you can get a list of zones where your tpu type is available [here](https://cloud.google.com/tpu/docs/regions-zones)

```
ZONE=us-central1-b
TPU_TYPE=v3-8
VM_NAME=ddpm

gcloud alpha compute tpus tpu-vm create $VM_NAME \
    --zone $ZONE \
    --accelerator-type $TPU_TYPE \
    --version tpu-vm-base

gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE -- \
```

When connected install JAX:

```
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Then install ddpm-flax and required libraries

```
git clone https://github.com/yiyixuxu/denoising-diffusion-flax.git
cd denoising-diffusion-flax/denoising_diffusion_flax
pip install einops
pip install wandb
pip install --upgrade clu

 ```

create a tmux session

```
tmux new -s ddpm

```

And finally start the training:


```
python3 main.py --workdir=./fashion-mnist --mode=train --config=configs/fashion_mnist.py 
```

## Examples (using default setting)

### cifar10

To run

```
python3 main.py --workdir=./cifar10 --mode=train --config=configs/cifar10.py 
```

W&B project page: [ddpm-flax-cifar10](https://wandb.ai/yiyixu/ddpm-flax-cifar10?workspace=user-yiyixu)


### fashion-mnist

To run

```
python3 main.py --workdir=./fashion-mnist --mode=train --config=configs/fashion_mnist.py 
```
W&B project page:  [ddpm-flax-fashion-mnist](https://wandb.ai/yiyixu/ddpm-flax-fashion-mnist?workspace=user-yiyixu)

### oxford_flowers102

To run
```
python3 main.py --workdir=./flower102--mode=train --config=configs/oxford102.py 
```

W&B project page: [ddpm-flax-flower102](https://wandb.ai/yiyixu/ddpm-flax-flower102?workspace=user-yiyixu)

## Dataset 

the script can run directly on any TensorFlow dataset, just set the configuration field `data.dataset` to the desired dataset name. You can update the field in configuration file directly or pass `--config.data.dataset=your_dataset_name` on command line to override it

you can also select different batch size and image size for your data. See more details on `config.data` in the example configuration files under `configs/` folder

## W&B Logging

It use Weights and Bias logging by default, if you don't already have an W&B acccount, you can sign up [here](https://wandb.ai/signup) - you will also be given option later to create an account when you run the script on comand line 

To disable W&B logging, you can override with `--config` flag on command line

```
python3 main.py --workdir=./fashion-mnist --mode=train --config=configs/fashion_mnist.py --config.wandb.log_train=False
```

you can also choose to log generated sample and model checkpoints, see more details on `config.wandb` in the example configuration files under `configs/` folder

## Predict x0

By default, we train our model to predict noise by modifying its parameterization, if you want to predict `x_0` directly from `x_t`, set `config.ddpm.pred_x0=True`; The authors of DDPM paper claimed that they it lead to worse sample quality in their experiments 

## Self-Conditioning

Self-Conditioning is a useful technique for improving diffusion models. In a typical diffusion sampling process, the model iteratively predict `x0` in order to gradually denoise the image, and the `x0` estimated from previous step is discard in the new step; with self-conditioning, the model will also take previously generated samples as input.

You read more about the technique in the paper [Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning](https://arxiv.org/abs/2208.04202)

to apply self-conditioning technique, set `config.ddpm.self_condition=True`;

## P2 Weighting

P2 (perception prioritized) weighting optimizes the weighting scheme of the training objective function to improve sample quality. It encourages the diffusion model to focus on recovering signals from highly corrupted data, where the model learns global and perceptually rich concepts. 

You can read more about P2 weighting in the [paper](https://arxiv.org/abs/2204.00227) and check out the git [repo](https://github.com/jychoi118/P2-weighting)

By default, we do not apply P2 weighting. However you can apply it by change the values of its hyperparameters: `config.ddpm.p2_loss_weight_gamma` and `config.ddpm.p2_loss_weight_k`; the paper recomend use `p2_loss_weight_gamma=1` and `p2_loss_weight_k=1`


## Model EMA 

By default, we will keep track of an EMA version of the model and use it to generate samples. You can find the list of hyperparameters for ema from `config.ema`



