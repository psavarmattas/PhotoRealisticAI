import numpy as np
from scipy.linalg import sqrtm
from skimage.transform import resize
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import image
from tqdm import tqdm

def calculate_fid(real_images, generated_images):
    """
    Calculate the Fréchet Inception Distance (FID) between real and generated images.

    Parameters:
    - real_images (numpy.ndarray): Real images with shape (n_samples, height, width, channels).
    - generated_images (numpy.ndarray): Generated images with shape (n_samples, height, width, channels).

    Returns:
    - float: The calculated FID.
    """

    def calculate_activation_statistics(images, model):
        """
        Calculate the activation statistics (mean and covariance) of images using a specified model.

        Parameters:
        - images (numpy.ndarray): Images with shape (n_samples, height, width, channels).
        - model (tf.keras.Model): Feature extraction model.

        Returns:
        - numpy.ndarray: Mean activation.
        - numpy.ndarray: Covariance activation.
        """
        activations = model.predict(images)
        mean = np.mean(activations, axis=0)
        covariance = np.cov(activations, rowvar=False)
        return mean, covariance

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
        """
        Calculate the Fréchet Distance between two multivariate Gaussians.

        Parameters:
        - mu1 (numpy.ndarray): Mean of the first Gaussian.
        - sigma1 (numpy.ndarray): Covariance matrix of the first Gaussian.
        - mu2 (numpy.ndarray): Mean of the second Gaussian.
        - sigma2 (numpy.ndarray): Covariance matrix of the second Gaussian.

        Returns:
        - float: The Fréchet Distance.
        """
        term1 = np.trace(sigma1 + sigma2 - 2 * sqrtm(sigma1 @ sigma2))
        term2 = np.linalg.norm(mu1 - mu2)
        return term1 + term2

    # Load InceptionV3 model for feature extraction
    inception_model = InceptionV3(include_top=False, weights="imagenet", pooling='avg', input_shape=(128, 128, 3))
    
    # Resize images to match InceptionV3 input size
    real_images_resized = np.array([resize(img, (128, 128, 3), mode='reflect', anti_aliasing=True) for img in real_images])
    generated_images_resized = np.array([resize(img, (128, 128, 3), mode='reflect', anti_aliasing=True) for img in generated_images])

    # Preprocess images for InceptionV3
    real_images_preprocessed = preprocess_input(real_images_resized)
    generated_images_preprocessed = preprocess_input(generated_images_resized)

    # Calculate activation statistics for real and generated images
    real_mean, real_covariance = calculate_activation_statistics(real_images_preprocessed, inception_model)
    generated_mean, generated_covariance = calculate_activation_statistics(generated_images_preprocessed, inception_model)

    # Calculate FID
    fid = calculate_frechet_distance(real_mean, real_covariance, generated_mean, generated_covariance)

    return fid