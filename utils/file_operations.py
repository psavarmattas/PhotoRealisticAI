import os

checkpoint_dir = "./ckpt"
generator_model_dir = "/generator"
discriminator_model_dir = "/discriminator"
plot_dir = "./plot/"
fig_dir = "./fig/"
out_dir = "./out_metrics/"

def creating_dirs():

    """
    Create necessary directories for storing checkpoints, plots, figures, and metrics.

    This function checks if the required directories exist and creates them if not.
    It creates subdirectories for generator and discriminator models within the checkpoint directory.

    Returns:
    None
    """
    
    print("====================================================================================================")
    print("Creating directories for storing checkpoints, plots, figures, and metrics...")
    print("====================================================================================================")

    # Prepare a directory to store all the checkpoints.

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        if not os.path.exists(checkpoint_dir+generator_model_dir):
            os.makedirs(checkpoint_dir+generator_model_dir)
        if not os.path.exists(checkpoint_dir+discriminator_model_dir):
            os.makedirs(checkpoint_dir+discriminator_model_dir)
    
    # Prepare a directory to store all the plots.
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # Prepare a directory to store all the figures.
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    # Prepare a directory to store all the metrics.
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def log_to_file(logText, metricLog):

    """
    Log text to different files based on the specified metric log type.

    This function takes log text and a metric log type as parameters and appends the log text
    to the corresponding log file (e.g., metric, verbose, epoch).

    Args:
        logText (str): The text to be logged.
        metricLog (int): The type of metric log (1: metric, 2: verbose, 3: epoch).

    Returns:
        None
    """

    if metricLog == 1:
        file_path = os.path.join(out_dir, 'logMetric.txt')
        if not os.path.isfile(file_path):
            with open(file_path, 'w') as file:
                pass
        else:
            with open(file_path, "a") as file:
                file.write("\n" + logText)
    elif metricLog == 2:
        file_path = os.path.join(out_dir, 'logVerbose.txt')
        if not os.path.isfile(file_path):
            with open(file_path, 'w') as file:
                pass
        else:
            with open(file_path, "a") as file:
                file.write("\n" + logText)
    elif metricLog == 3:
        file_path = os.path.join(out_dir, 'logEpoch.txt')
        if not os.path.isfile(file_path):
            with open(file_path, 'w') as file:
                pass
        else:
            with open(file_path, "a") as file:
                file.write("\n" + logText)        
    else:
        print("Invalid Metric Log")