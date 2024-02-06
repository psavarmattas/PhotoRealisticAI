from keras.optimizers import Adam
from keras.models import Sequential
from utils.multi_GPU import strategy

def build_gan(generator, discriminator):

    """
    Build and compile a GAN (Generative Adversarial Network) model.

    Parameters:
    - generator (tf.keras.models.Model): The generator model.
    - discriminator (tf.keras.models.Model): The discriminator model.

    Returns:
    - tf.keras.models.Model: Compiled GAN model.

    The GAN model is created by stacking the generator on top of the discriminator. During training,
    the discriminator is set as non-trainable to prevent its weights from updating. The GAN is trained
    to generate realistic images that can fool the discriminator.

    Example:
    ```python
    # Assuming you have already built the generator and discriminator models
    gan_model = build_gan(generator_model, discriminator_model)
    gan_model.summary()
    ```

    Note:
    - Ensure that the generator and discriminator models are already built and compiled.
    - Adjust the learning_rate and beta_1 parameters in the Adam optimizer based on your requirements.
    """

    if strategy is not None:
        with strategy.scope():
            # Setting discriminator as non-trainable, so its weights won't update when training the GAN
            discriminator.trainable = False

            # Creating the GAN model
            model = Sequential()
                
            # Adding the generator
            model.add(generator)
                
            # Adding the discriminator
            model.add(discriminator)

            # Compiling the GAN model
            optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    else:
        # Setting discriminator as non-trainable, so its weights won't update when training the GAN
        discriminator.trainable = False

        # Creating the GAN model
        model = Sequential()
            
        # Adding the generator
        model.add(generator)
            
        # Adding the discriminator
        model.add(discriminator)

        # Compiling the GAN model
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model