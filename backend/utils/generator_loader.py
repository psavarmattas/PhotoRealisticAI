import os
import tensorflow as tf
from utils.file_operations import checkpoint_dir, generator_model_dir
from models.generator import build_generator


def generator_loader(noise_dimension):

    """
    Make or restore the generator model.

    Either restores the latest model checkpoint or creates a fresh one if no checkpoint is available.

    Args:
        noise_dimension (int): The dimension of the noise input.

    Returns:
        tf.keras.Model: The generator model.

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """

    noise_dimension = noise_dimension
    latest_checkpoint = tf.train.latest_checkpoint(os.path.join(checkpoint_dir + generator_model_dir))
    if latest_checkpoint is not None:
        print(">>>Generator version:", latest_checkpoint)
        generator_model = build_generator(noise_dimension)
        checkpoint = tf.train.Checkpoint(model=generator_model)
        checkpoint.restore(latest_checkpoint)
        return generator_model
    else:
        raise FileNotFoundError("No checkpoint found for the generator model.")