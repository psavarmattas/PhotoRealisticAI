from PIL import Image
import numpy as np
from utils.data_generator import generateNoiseSamples

def generateNewImageFromGeneratorAI(generator):

    """
    Generate a single image using the provided generator model.

    Parameters:
    - generator (tf.keras.Model): The generator model used to generate the image.

    Returns:
    - PIL.Image.Image: The generated image.
    """

    # Generate noise samples
    noise_dim = 100
    X_noise = generateNoiseSamples(1, noise_dim)

    # Use generator to create an image
    generated_image = generator.predict(X_noise)
    # Rescale pixel values to the range [0, 255]
    generated_image = ((0.5 * generated_image + 0.5) * 255).astype(np.uint8)

    # Convert NumPy array to PIL Image
    generated_image_pil = Image.fromarray(generated_image[0])

    return generated_image_pil