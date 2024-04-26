from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, LeakyReLU, Dropout, Flatten
from utils.multi_GPU import strategy

def build_discriminator(image_shape=(128, 128, 3)):
    """
    Build and compile a discriminator model for a GAN (Generative Adversarial Network).

    Parameters:
    - image_shape (tuple): The shape of input images (height, width, channels). Defaults to (128, 128, 3) for typical RGB images.

    Returns:
    - tf.keras.models.Model: Compiled discriminator model.

    The discriminator architecture consists of convolutional layers with LeakyReLU activation,
    followed by dropout and a dense layer for binary classification. The model is trained
    to distinguish between real and generated images.

    Example:
    ```python
    discriminator_model = build_discriminator(image_shape=(256, 256, 3))
    discriminator_model.summary()
    ```

    Note:
    The default image_shape is suitable for typical RGB images. Adjust image_shape according
    to the dimensions of the images in your dataset.
    """
    if strategy is not None:
        with strategy.scope():
                model = Sequential()
                # Initial convolutional layer
                model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=image_shape))
                model.add(LeakyReLU(0.2))
                
                # Second convolutional layer
                model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
                model.add(LeakyReLU(0.2))
                
                # Third convolutional layer
                model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
                model.add(LeakyReLU(0.2))
                
                # Fourth convolutional layer
                model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
                model.add(LeakyReLU(0.2))
                
                # Fifth convolutional layer
                model.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
                model.add(LeakyReLU(0.2))

                # Flatten and dense layer for classification
                model.add(Flatten())
                model.add(Dropout(0.4))
                model.add(Dense(1, activation='sigmoid'))
                
                # Define optimizer and compile model
                optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    else:
        model = Sequential()
        
        # Initial convolutional layer
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=image_shape))
        model.add(LeakyReLU(0.2))
        
        # Second convolutional layer
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(0.2))
        
        # Third convolutional layer
        model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(0.2))
        
        # Fourth convolutional layer
        model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(0.2))
        
        # Fifth convolutional layer
        model.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(0.2))

        # Flatten and dense layer for classification
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        
        # Define optimizer and compile model
        optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model