from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU, Reshape

def build_generator(latent_dim, channels=3):

    """
    Build and compile a generator model for a GAN (Generative Adversarial Network).

    Parameters:
    - latent_dim (int): The dimensionality of the latent space.
    - channels (int): Number of image channels. Defaults to 3 for typical RGB images.

    Returns:
    - tf.keras.models.Model: Compiled generator model.

    The generator architecture consists of dense and deconvolutional layers with LeakyReLU activation,
    producing an image with the specified number of channels. The model is trained to generate images
    that resemble real images.

    Example:
    ```python
    generator_model = build_generator(latent_dim=100, channels=3)
    generator_model.summary()
    ```

    Note:
    - Adjust the latent_dim parameter based on the desired dimensionality of the latent space.
    - The channels parameter specifies the number of image channels (e.g., 3 for RGB images).
    - The 'tanh' activation in the output layer scales the generated image pixel values to the range [-1, 1].
    """

    model = Sequential()
    
    # Initial dense layer
    model.add(Dense(32 * 32 * 128, input_dim=latent_dim))
    model.add(LeakyReLU(0.2))
    
    # Reshape to (32, 32, 128) tensor for convolutional layers
    model.add(Reshape((32, 32, 128)))
    
    # First deconvolutional layer
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Second deconvolutional layer
    model.add(Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Third deconvolutional layer
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Fourth deconvolutional layer
    model.add(Conv2DTranspose(64, (4, 4), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(0.2))
    
    # Output convolutional layer with 'tanh' activation
    model.add(Conv2D(channels, (8, 8), activation='tanh', padding='same'))

    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model
