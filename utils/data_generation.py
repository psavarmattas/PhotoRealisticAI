from numpy import zeros, ones
from numpy.random import randn, randint

def generate_real_samples(dataset, num_samples):

    """
    Generate real samples from the dataset for training a DCGAN.

    Randomly selects a specified number of samples from the dataset.

    Args:
        dataset (numpy.ndarray): The dataset of real images.
        num_samples (int): The number of real samples to generate.

    Returns:
        tuple: A tuple containing the real samples (X) and their corresponding labels (y).
    """

    sample_indices = randint(0, dataset.shape[0], num_samples)
    X = dataset[sample_indices]
    y = ones((num_samples, 1))
    return X, y

def generate_noise_samples(num_samples, noise_dim):

    """
    Generate noise samples for training a DCGAN.

    Generates random noise samples with a specified dimension.

    Args:
        num_samples (int): The number of noise samples to generate.
        noise_dim (int): The dimension of the noise samples.

    Returns:
        numpy.ndarray: The generated noise samples.
    """

    X_noise = randn(noise_dim * num_samples)
    X_noise = X_noise.reshape(num_samples, noise_dim)
    return X_noise 

def generate_fake_samples(generator, noise_dim, num_samples):

    """
    Generate fake samples using a generator model for training a DCGAN.

    Generates fake samples by first generating noise and passing it through the generator.

    Args:
        generator (tensorflow.keras.Model): The generator model.
        noise_dim (int): The dimension of the noise samples.
        num_samples (int): The number of fake samples to generate.

    Returns:
        tuple: A tuple containing the generated fake samples (X) and their corresponding labels (y).
    """

    X_noise = generate_noise_samples(num_samples, noise_dim)
    X = generator.predict(X_noise)
    y = zeros((num_samples, 1 ))
    return X, y
