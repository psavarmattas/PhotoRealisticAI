import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from utils.file_operations import checkpoint_dir, discriminator_model_dir, generator_model_dir
from models.discriminator import build_discriminator
from models.generator import build_generator

def make_or_restore_model_discriminator():

    """
    Make or restore the discriminator model.

    Either restores the latest model checkpoint or creates a fresh one if no checkpoint is available.

    Returns:
        tf.keras.Model: The discriminator model.
    """

    latest_checkpoint = tf.train.latest_checkpoint(os.path.join(checkpoint_dir + discriminator_model_dir))
    if latest_checkpoint is not None:
        print("====================================================================================================")
        print("Restoring Discriminator from", latest_checkpoint)
        print("====================================================================================================")
        discriminator_model = build_discriminator()
        checkpoint = tf.train.Checkpoint(model=discriminator_model)
        checkpoint.restore(latest_checkpoint)
        print("====================================================================================================")
        return discriminator_model
    else: 
        print("====================================================================================================")
        print("Creating a new Discriminator model")
        print("====================================================================================================")
        discriminator_model = build_discriminator()
        print("====================================================================================================")
        return discriminator_model

def save_discriminator_checkpoint(discriminator_model, epoch):

    """
    Save the discriminator model using checkpoints.

    Args:
        discriminator_model (tf.keras.Model): The discriminator model to save.
        epoch (int): The current epoch number.

    Returns:
        None
    """

    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M")
    checkpoint_prefix = os.path.join(checkpoint_dir + discriminator_model_dir, f"discriminator-ckpt-{current_time}")
    checkpoint = tf.train.Checkpoint(model=discriminator_model)
    checkpoint.save(file_prefix=checkpoint_prefix + "-{}".format(epoch))

def make_or_restore_model_generator(noise_dimension):

    """
    Make or restore the generator model.

    Either restores the latest model checkpoint or creates a fresh one if no checkpoint is available.

    Args:
        noise_dimension (int): The dimension of the noise input.

    Returns:
        tf.keras.Model: The generator model.
    """

    noise_dimension = noise_dimension
    latest_checkpoint = tf.train.latest_checkpoint(os.path.join(checkpoint_dir + generator_model_dir))
    if latest_checkpoint is not None:
        print("====================================================================================================")
        print("Restoring Generator from", latest_checkpoint)
        print("====================================================================================================")
        generator_model = build_generator(noise_dimension)
        checkpoint = tf.train.Checkpoint(model=generator_model)
        checkpoint.restore(latest_checkpoint)
        print("====================================================================================================")
        return generator_model
    else:
        print("====================================================================================================")
        print("Creating a new Generator model")
        print("====================================================================================================")
        generator_model = build_generator(noise_dimension)
        print("====================================================================================================")
        return generator_model

def save_generator_checkpoint(generator_model, epoch):

    """
    Save the generator model using checkpoints.

    Args:
        generator_model (tf.keras.Model): The generator model to save.
        epoch (int): The current epoch number.

    Returns:
        None
    """

    current_time = datetime.now().strftime("%d-%m-%Y-%H-%M")
    checkpoint_prefix = os.path.join(checkpoint_dir + generator_model_dir, f"generator-ckpt-{current_time}")
    checkpoint = tf.train.Checkpoint(model=generator_model)
    checkpoint.save(file_prefix=checkpoint_prefix + "-{}".format(epoch))
