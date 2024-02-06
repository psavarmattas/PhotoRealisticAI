import tensorflow as tf

strategy = None

def multi_GPU_trainer():
    """
    Prompt the user to choose between single-GPU and multi-GPU training.

    This function interacts with the user to determine whether they want to train
    the model using multiple GPUs (if available) or a single GPU.

    Sets the global variable `strategy` to a `tf.distribute.MirroredStrategy` if
    multi-GPU training is chosen, or to `None` for single-GPU training.

    Returns:
        None
    """
    global strategy
    
    print("====================================================================================================")
    multi_GPU = input(str("Do you want to train with Multi-GPU? (Y/N) "))
    print("====================================================================================================")
    while True:
        if ((multi_GPU == "Y") or (multi_GPU == "y")):
            strategy = tf.distribute.MirroredStrategy()
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
            break
        elif ((multi_GPU == "N") or (multi_GPU == "n")):
            print("Training with a single GPU.")
            strategy = None
            break
        else:
            print("Invalid input. Please try again.")
            strategy = None