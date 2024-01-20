import tensorflow as tf

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
    print("\n", tf.config.list_physical_devices('CPU'))
    print("\nNum CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
    print("\n", tf.config.list_physical_devices('GPU'))
    print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))