"""Main file for running denoising-diffusion-flax.
"""
from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf


import train, sampling


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training or sampling hyperparameter configuration.',
    lock_config=True)

flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("mode", "train", "Running mode: train or sample")

def main(argv):
  if len(argv) > 2:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}', 
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  if FLAGS.mode == "train":
      train.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "sample":
      sampling.sample(FLAGS.config, FLAGS.workdir)
  else:
      raise ValueError(f"Mode {FLAGS.mode} not recognized.")

if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)