import warnings
warnings.filterwarnings('ignore')

import time
import sys
from utils.data_processing import *
from utils.model_operations import *
from models.gan import *
from utils.train import *
from utils.check_tf import *

def train_or_load(train_model):

    """
    Perform training or loading of GAN models based on user input.

    Args:
        train_model (str): User input indicating whether to train (Y/y) or load (N/n) the models.

    Returns:
        None
    """

    # Preprocessing and loading dataset
    dataset = load_and_preprocess_dataset()
    if ((train_model=="Y") or (train_model=="y")):
        # Set Arguments for training
        noise_dimension = 100
        num_epochs=50
        batch_size=64
        display_frequency=1
        verbose=1
        saved_images = []

        # Build or restore discriminator model
        discriminator = make_or_restore_model_discriminator()
        print("====================================================================================================")
        # Show discriminator model summary
        discriminator.summary()
        print("====================================================================================================")
        time.sleep(5)
        # Build or restore generator model
        generator = make_or_restore_model_generator(noise_dimension)
        print("====================================================================================================")
        # Show generator model summary
        generator.summary()
        time.sleep(5)
        print("====================================================================================================")
        # Build GAN model from generator and discriminator
        gan_model = build_gan(generator, discriminator)
        train(generator, discriminator, gan_model, dataset, noise_dimension, num_epochs, batch_size, display_frequency, verbose)
        print("====================================================================================================")
        print("Training of the GAN model on the dataset is completed")
        print("====================================================================================================")
        load_or_finish=str(input("Do you want to load the model or end training? (Y/N): "))
        print("====================================================================================================")
        if ((load_or_finish=="Y") or (load_or_finish=="y")):
            # Load generator model
            noise_dimension = 100
            generator = make_or_restore_model_generator(noise_dimension)
            generator.summary()
            print("====================================================================================================")
        elif ((load_or_finish=="N") or (load_or_finish=="n")):
            sys.exit()
    elif ((train_model=="N") or (train_model=="n")):
        # Build or restore generator model
        noise_dimension = 100
        generator = make_or_restore_model_generator(noise_dimension)
        print("====================================================================================================")
        # Show generator model summary
        generator.summary()
        time.sleep(5)
        print("====================================================================================================")
        print("Model is loaded successfully")
        print("====================================================================================================")