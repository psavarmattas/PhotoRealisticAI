import warnings
warnings.filterwarnings('ignore')

import time
import sys
from utils.data_processing import *
from utils.model_operations import *
from models.gan import *
from utils.train import *
from utils.check_tf import *

def train_or_load():

    """
    Perform training or loading of GAN models based on user input.

    Returns:
        tf.keras.Model or None: The generator model if successfully loaded or trained, otherwise None.
    """

    while True:
        # Asking the user if they want to train the model or not.
        print("====================================================================================================")
        train_model = str(input("Do you want to train the model? (Y/N): "))
        print("====================================================================================================")
        
        if ((train_model=="Y") or (train_model=="y")):
            # Preprocessing and loading dataset
            dataset = load_and_preprocess_dataset(load_limit=202599)
            # Set Arguments for training
            noise_dimension = 100
            num_epochs=20
            batch_size=64
            display_frequency=1
            verbose=1
            fid_frequency=10
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
            train(generator, discriminator, gan_model, dataset, noise_dimension, num_epochs, batch_size, display_frequency, verbose, fid_frequency)
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
                return generator
            elif ((load_or_finish=="N") or (load_or_finish=="n")):
                sys.exit()
            else:
                print("Invalid input. Please try again.")
        elif ((train_model=="N") or (train_model=="n")):
            dataset = load_and_preprocess_dataset(load_limit=50000)
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
            fid_history = []
            real_images_fid, real_labels_fid = generate_real_samples(dataset, 1000)
            generated_images_fid, generated_label_fid = generate_fake_samples(generator, noise_dimension, len(real_images_fid))
            fid_value = calculate_fid(real_images_fid, generated_images_fid)
            fid_history.append(fid_value)
            print(f"FID at last checkpoint: {fid_value}")
            print("====================================================================================================")
            return generator
        else:
            print("Invalid input. Please try again.")