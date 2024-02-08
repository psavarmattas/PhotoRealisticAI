from utils.model_operations import save_discriminator_checkpoint, save_generator_checkpoint
from utils.data_generation import generate_real_samples, generate_fake_samples, generate_noise_samples
from utils.visualization import generate_images
from utils.image_plotting import plot_generated_images
from utils.metric_plotting import plot_training_metrics, plot_fid_score
from utils.calculate_fid import calculate_fid
from utils.cooldown_gpu import cooldown
from utils.file_operations import log_to_file
import time, sys
import numpy as np

def train(generator_model, discriminator_model, gan_model, dataset, noise_dimension,
            num_epochs, batch_size, display_frequency, verbose, fid_frequency):

    """
    Train the GAN model on the given dataset.

    Args:
        generator_model (tf.keras.Model): The generator model.
        discriminator_model (tf.keras.Model): The discriminator model.
        gan_model (tf.keras.Model): The GAN model.
        dataset (numpy.ndarray): The dataset for training.
        noise_dimension (int): The dimension of the noise input.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        display_frequency (int): Frequency to display generated images.
        verbose (bool): If True, print training details for each batch.
        fid_frequency (int): Frequency to calculate FID.

    Returns:
        None
    """

    print("====================================================================================================")
    print("Training the GAN model on the dataset and get the saved images list")
    print("====================================================================================================")

    # We include the current epoch in the folder name.
    
    # Create lists to store metrics for plotting
    discriminator_loss_history = []
    discriminator_accuracy_history = []
    generator_loss_history = []
    fid_history = []
    
    # Create an empty list to store generated images for each epoch
    saved_images_for_epochs = []
    saved_images = []
    
    # Calculate the number of batches per epoch
    batches_per_epoch = int(dataset.shape[0] / batch_size)
    
    # Calculate half the size of a batch
    half_batch_size   = int(batch_size / 2)
    
    # Setting cooldown time for GPU cooldown
    cooldown_duration_minutes=5
    
    # Loop over all epochs
    for epoch in range(num_epochs):
        train_step(generator_model, discriminator_model, gan_model, dataset, noise_dimension, 
                    epoch, batches_per_epoch, half_batch_size, batch_size, verbose,
                    discriminator_loss_history, discriminator_accuracy_history, generator_loss_history)
        
        # Save models after every epoch
        save_discriminator_checkpoint(discriminator_model, epoch)
        save_generator_checkpoint(generator_model, epoch)
        
        # Display generated images at the specified frequency
        if epoch % display_frequency == 0:
            generated_images_for_epoch = generate_images(epoch, generator_model)
            saved_images_for_epochs.append(generated_images_for_epoch)

            # Plot generated images to visualize the progress of the generator
            plot_generated_images(epoch, generator_model)

            # Plot the training metrics
            plot_training_metrics(discriminator_loss_history, discriminator_accuracy_history, generator_loss_history, epoch+1)
            # resetting lists to store metrics for plotting for next batch
            discriminator_loss_history = []
            discriminator_accuracy_history = []
            generator_loss_history = []

        if epoch % fid_frequency == 0:
            real_images_fid, real_labels_fid = generate_real_samples(dataset, 1000)
            generated_images_fid, generated_label_fid = generate_fake_samples(generator_model, noise_dimension, len(real_images_fid))
            fid_value = calculate_fid(real_images_fid, generated_images_fid)
            fid_history.append(fid_value)
            print(f"FID at Epoch {epoch+1}: {fid_value}")
            logText=f"FID at Epoch {epoch+1}: {fid_value}"
            log_to_file(logText, 4)
            plot_fid_score(fid_history, epoch+1)
            print("====================================================================================================")

        time.sleep(10)

        # Cooldown after every 5 epochs for 5 minutes
        if (epoch + 1) % 5 == 0:  # Cooldown after every 5 epochs
            cooldown(cooldown_duration_minutes)

    saved_images_for_epochs = saved_images

def train_step(generator_model, discriminator_model, gan_model, dataset, noise_dimension,
                epoch, batches_per_epoch, half_batch_size,  batch_size, verbose,
                discriminator_loss_history, discriminator_accuracy_history, generator_loss_history):
        """
        Perform a single training step within an epoch.

        Args:
            generator_model (tf.keras.Model): The generator model.
            discriminator_model (tf.keras.Model): The discriminator model.
            gan_model (tf.keras.Model): The GAN model.
            dataset (numpy.ndarray): The dataset for training.
            noise_dimension (int): The dimension of the noise input.
            epoch (int): The current epoch.
            batches_per_epoch (int): The number of batches per epoch.
            half_batch_size (int): Half the size of a batch.
            batch_size (int): The batch size for training.
            verbose (bool): If True, print training details for each batch.
            discriminator_loss_history (list): List to store discriminator loss.
            discriminator_accuracy_history (list): List to store discriminator accuracy.
            generator_loss_history (list): List to store generator loss.

        Returns:
            None
        """
        # Record start time for the epoch
        epoch_start_time = time.time()
        
        # Loop over all batches within this epoch
        for batch_num in range(batches_per_epoch):
            
            # Generate a batch of real images and their corresponding labels
            real_images, real_labels    = generate_real_samples(dataset, half_batch_size)
            # Train the discriminator on the real images and calculate loss and accuracy
            dsr_loss_real, dsr_acc_real = discriminator_model.train_on_batch(real_images, real_labels)

            # Generate a batch of fake images and their corresponding labels
            fake_images, fake_labels    = generate_fake_samples(generator_model, noise_dimension, half_batch_size)
            # Train the discriminator on the fake images and calculate loss and accuracy
            dsr_loss_fake, dsr_acc_fake = discriminator_model.train_on_batch(fake_images, fake_labels)
            
            # Calculate the average discriminator loss and accuracy over real and fake images
            dsr_loss = 0.5 * (dsr_loss_real + dsr_loss_fake)

            dsr_acc = 0.5 * (dsr_acc_real + dsr_acc_fake)

            # Generate noise samples and their corresponding labels for training the generator
            gan_noise  = generate_noise_samples(batch_size, noise_dimension)
            gan_labels = np.ones((batch_size, 1))

            # Train the generator and calculate loss
            gen_loss, _ = gan_model.train_on_batch(gan_noise, gan_labels)

            if verbose:  # This condition checks if verbose is non-zero
                # Log training information for this batch
                logMetric=f"{epoch+1}, {batch_num+1}, {dsr_loss:.6f}, {100*dsr_acc:.2f}, {gen_loss:.6f}"
                log_to_file(logMetric, 1)
                logText= f"[ Epoch: {epoch+1} , Batch: {batch_num+1} ] --> [ Discriminator Loss : {dsr_loss:.6f} , Discriminator Accuracy: {100*dsr_acc:.2f}% ] [ Generator Loss: {gen_loss:.6f} ]"
                log_to_file(logText, 2)
                print(f"[ Epoch: {epoch+1} , Batch: {batch_num+1} ] --> [ Discriminator Loss : {dsr_loss:.6f} , Discriminator Accuracy: {100*dsr_acc:.2f}% ] [ Generator Loss: {gen_loss:.6f} ]")

            # Collect metrics for plotting
            discriminator_loss_history.append(dsr_loss)
            discriminator_accuracy_history.append(100 * dsr_acc)
            generator_loss_history.append(gen_loss)

            if (dsr_loss >= 15.0) | (gen_loss >= 15.0):
                print("====================================================================================================")
                print("Loss is too high! Training is stopped.")
                print("\nNOTE: Please delete the last checkpoint file & write the name of the previous checkpoint .index file to the checkpoint file in the generator & discriminator checkpoint folders.")
                print("====================================================================================================")
                sys.exit(1)
            else:
                continue

        # Log Epoch count
        logEpoch=f"{epoch+1}"

        log_to_file(logEpoch, 3)
        # Record end time for the epoch
        epoch_end_time = time.time()

        # Calculate and print the time taken for the epoch
        epoch_duration_seconds = epoch_end_time - epoch_start_time
        hours, remainder = divmod(epoch_duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print("====================================================================================================")
        print(f"Epoch {epoch+1} took {int(hours):02}:{int(minutes):02}:{seconds:.2f}")
        print("====================================================================================================")