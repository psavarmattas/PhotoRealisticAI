import tensorflow as tf
import sys

def check_tf():

    """
    Check if TensorFlow is installed and detect available CPUs and GPUs.

    This function prints information about the TensorFlow version, the list of available CPUs,
    the number of available CPUs, the list of available GPUs, and the number of available GPUs.

    Returns:
    None
    """

    print("====================================================================================================")
    print("Checking if Tensorflow is installed & find the CPUs & GPUs that are available...")
    print("====================================================================================================")

    print("Tensorflow Version Available: ", tf.version.VERSION)
    if tf.version.VERSION >= '2.0':
        if len(tf.config.list_physical_devices('GPU')) >= 2:
            print("\n", tf.config.list_physical_devices('GPU'))
            print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        elif len(tf.config.list_physical_devices('CPU')) >= 2:
            print("\n", tf.config.list_physical_devices('CPU'))
            print("\nNum CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
            print("\nNo GPUs found therefore training will be done using CPU.")
        else:
            print("\n>>> No GPU/CPU found. Please check if TensorFlow and it's addons are installed correctly.")
            sys.exit(1)
    else:
        print("The current Tensorflow version is less than v2.0. Please upgrade to v2.0 or higher to continue.")
        sys.exit(1)