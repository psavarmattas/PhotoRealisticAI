from utils.data_generation import generate_noise_samples
from utils.file_operations import plot_dir
from matplotlib import pyplot as plt
import os

def plot_generated_images(epoch, generator, num_samples=6, noise_dim=100, figsize=(15, 3)):

    """
    Plot and visualize generated images from the generator model for a given epoch.

    Args:
        epoch (int): The epoch number for which images are being generated and plotted.
        generator (tensorflow.keras.Model): The generator model used to generate images.
        num_samples (int, optional): The number of generated images to plot. Defaults to 6.
        noise_dim (int, optional): The dimension of the noise samples used for generation. Defaults to 100.
        figsize (tuple, optional): The size of the figure for plotting. Defaults to (15, 3).
    """
    
    # Generate noise samples
    X_noise = generate_noise_samples(num_samples, noise_dim)
    
    # Use generator to produce images from noise
    X = generator.predict(X_noise, verbose=0)

    # Rescale images to [0, 1] for visualization
    X = (X + 1) / 2

    # Plotting the images
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)

    for i in range(num_samples):
        axes[i].imshow(X[i])
        axes[i].axis('off')

    # Add a descriptive title
    fig.suptitle(f"Generated Images at Epoch {epoch+1}", fontsize=22)
    plt.tight_layout()
        
    # save figure
    image_path = os.path.join((plot_dir+"epoch_"+(str(epoch+1))))
    plt.savefig(fname=image_path)
    
    # Uncomment the line below if you want to display the plot
    # plt.show()

