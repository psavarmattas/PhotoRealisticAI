from utils.check_dependencies import *
from utils.check_tf import *
from utils.train_load_finish import *
from utils.file_operations import *

def main():

    """
    Check dependencies, TensorFlow setup, and initiate the model training or loading based on user input.

    This script checks for all necessary dependencies, verifies the TensorFlow installation,
    and prompts the user to choose between training the model or loading an existing one.

    Returns:
        None
    """

    # Check all dependencies
    check_dependencies()

    # Checking TensorFlow and its dependencies
    check_tf()
    
    # Making directories for everything
    creating_dirs()

    # Asking the user if they want to train the model or load it.
    generator=train_or_load()
    generateNewImageFromGAN(generator)

if __name__ == "__main__":
    main()