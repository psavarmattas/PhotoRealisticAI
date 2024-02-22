from numpy.random import randn

def generateNoiseSamples(num_samples, noise_dim):

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