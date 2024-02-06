from utils.check_dependencies import check_dependencies
from utils.check_tf import check_tf
from utils.train_load_finish import train_or_load
from utils.file_operations import creating_dirs
from utils.multi_GPU import multi_GPU_trainer
from utils.visualization import generateNewImageFromGAN

def main():
    """
    Check dependencies, TensorFlow setup, and initiate the model training or loading based on user input.

    This script performs a series of checks to ensure that all necessary dependencies are installed,
    verifies the correct installation of TensorFlow, and gives the user the option to either train a
    new model or load an existing one. It also prompts the user to choose between multi-GPU and single-GPU
    training setups.

    Returns:
        None
    """
    # Check all dependencies
    #check_dependencies()

    # Checking TensorFlow and its dependencies
    #check_tf()
    
    # Asking user to select multi GPU or single GPU
    multi_GPU_trainer()
    
    # # Making directories for everything
    creating_dirs()

    # # Asking the user if they want to train the model or load it.
    generator=train_or_load()
    generateNewImageFromGAN(generator)

if __name__ == "__main__":
    main()