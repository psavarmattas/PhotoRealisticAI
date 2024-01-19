from utils.data_generation import *
from matplotlib import pyplot as plt

def generate_images(epoch, generator, num_samples=6, noise_dim=100):

    """
    Generate images from the generator model for a given epoch.

    Args:
        epoch (int): The epoch number for which images are being generated.
        generator (tf.keras.Model): The generator model.
        num_samples (int, optional): Number of images to generate. Defaults to 6.
        noise_dim (int, optional): Dimension of the noise input. Defaults to 100.

    Returns:
        numpy.ndarray: An array of generated images with shape (num_samples, height, width, channels).
    """

    # Generate noise samples
    X_noise = generate_noise_samples(num_samples, noise_dim)
    
    # Use generator to produce images from noise
    X = generator.predict(X_noise, verbose=0)

    # Rescale images to [0, 1] for visualization
    X = (X + 1) / 2

    return X

def generateNewImageFromGAN(generator):

    """
    Generate a single image using the generator model.

    Args:
        generator (tf.keras.Model): The generator model.

    Returns:
        None
    """

    # Generate noise samples
    noise_dim = 100
    X_noise = generate_noise_samples(1, noise_dim)

    # Use generator to create an image
    generated_image = generator.predict(X_noise)

    # Rescale pixel values to the range [0, 1]
    generated_image = 0.5 * generated_image + 0.5

    # Display the generated image
    plt.imshow(generated_image[0])
    plt.axis('off')
    plt.show()