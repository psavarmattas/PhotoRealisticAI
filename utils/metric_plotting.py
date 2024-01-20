from matplotlib import pyplot as plt
from utils.file_operations import *

def plot_training_metrics(discriminator_loss, discriminator_accuracy, generator_loss, epoch):

    """
    Plot training metrics, including Discriminator Loss, Discriminator Accuracy, Generator Loss,
    and their combinations, for a given epoch.

    Args:
        discriminator_loss (list): List of Discriminator loss values across training iterations.
        discriminator_accuracy (list): List of Discriminator accuracy values across training iterations.
        generator_loss (list): List of Generator loss values across training iterations.
        epoch (int): The epoch number for which metrics are being plotted.

    Note:
        The function creates separate plots for Discriminator Accuracy, Discriminator Loss, 
        Generator Loss, Discriminator Loss vs. Generator Loss, and a combined plot with all metrics.

    Returns:
        None
    """

    iterations = range(len(discriminator_loss))

    # Create separate plots for each metric

    # Create a plot with Discriminator Accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(iterations, discriminator_accuracy, label='Discriminator Accuracy')
    plt.title('Discriminator Accuracy at '+str(epoch))
    plt.xlabel('Iteration')
    plt.legend()
    image_path = os.path.join(out_dir, 'discriminator_accuracy_at-'+str(epoch)+'.png')
    plt.savefig(fname=image_path)  # Save separately
    plt.close()

    # Create a plot with Discriminator Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(iterations, discriminator_loss, label='Discriminator Loss')
    plt.title('Discriminator Loss at '+str(epoch))
    plt.xlabel('Iteration')
    plt.legend()
    image_path = os.path.join(out_dir, 'discriminator_loss_at-'+str(epoch)+'.png')
    plt.savefig(fname=image_path)  # Save separately
    plt.close()

    # Create a plot with Generator Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(iterations, generator_loss, label='Generator Loss')
    plt.title('Generator Loss at '+str(epoch))
    plt.xlabel('Iteration')
    plt.legend()
    image_path = os.path.join(out_dir, 'generator_loss_at-'+str(epoch)+'.png')
    plt.savefig(fname=image_path)  # Save separately
    plt.close()
    
    # Create a plot with Discriminator Loss vs. Generator Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(iterations, discriminator_loss, label='Discriminator Loss')
    plt.plot(iterations, generator_loss, label='Generator Loss')
    plt.title('Discriminator Loss vs Generator Loss at '+str(epoch))
    plt.xlabel('Iteration')
    plt.legend()
    image_path = os.path.join(out_dir, 'discriminator_loss_vs_generator_loss-'+str(epoch)+'.png')
    plt.savefig(fname=image_path)  # Save separately
    plt.close()

    # Create a plot with all metrics together
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.plot(iterations, discriminator_loss, label='Discriminator Loss')
    plt.title('Discriminator Loss')
    plt.xlabel('Iteration')
    plt.legend()

    plt.subplot(132)
    plt.plot(iterations, discriminator_accuracy, label='Discriminator Accuracy')
    plt.title('Discriminator Accuracy')
    plt.xlabel('Iteration')
    plt.legend()

    plt.subplot(133)
    plt.plot(iterations, generator_loss, label='Generator Loss')
    plt.title('Generator Loss')
    plt.xlabel('Iteration')
    plt.legend()

    plt.tight_layout()

    # Save the combined plot
    image_path = os.path.join(out_dir, 'training_metrics_combined-'+str(epoch)+'.png')
    plt.savefig(fname=image_path)

    # Uncomment the line below if you want to display the plot
    # plt.show()

def plot_fid_score(fid_history, epoch):
    """
    Plot FID score history over training iterations and save the plot.

    Args:
        fid_history (list): List of FID scores at different iterations.
        epoch (int): The epoch number for which the FID is plotted.
        out_dir (str): The directory to save the plot. Defaults to the current working directory.

    Returns:
        None
    """
    iterations = range(len(fid_history))
    
    # Create a plot with FID Score History
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.plot(iterations, fid_history, label='FID Score')
    plt.title('FID at '+str(epoch))
    plt.xlabel('Epochs')
    plt.legend()
    image_path = os.path.join(out_dir, 'FID_at-'+str(epoch)+'.png')
    plt.savefig(fname=image_path)  # Save separately
    plt.close()