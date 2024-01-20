import tensorflow as tf
import glob
import os
from tqdm import tqdm
import numpy as np

# Define a function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(128,128)):

    """
    Load and preprocess an image from the specified path.

    This function reads an image from the given path, performs cropping and resizing,
    and normalizes pixel values to the range [-1, 1].

    Args:
        image_path (str): The path to the image file.
        target_size (tuple): The target size for resizing the image. Default is (128, 128).

    Returns:
        tf.Tensor: The preprocessed image as a TensorFlow tensor.
    """

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.crop_to_bounding_box(image, 20, 0, 178, 178)  # Crop 20 pixels from top
    # image = tf.image.resize(image, target_size)
    # image = (image / 127.5) - 1.0  # Normalize to the range [-1, 1]
    return image

def load_and_preprocess_dataset(load_limit):

    """
    Load and preprocess a dataset of images.

    This function reads a list of image paths from a specified directory, limits the number of images,
    and creates a TensorFlow dataset. It then applies the `load_and_preprocess_image` function
    to each image in parallel using tf.data.

    Returns:
        np.ndarray: The preprocessed dataset as a NumPy array.
    """

    # Define the directory of your images
    dataset_dir = "./dataset/celebA_dataset"

    # Get a list of all image paths in the directory
    image_paths = glob.glob(os.path.join(dataset_dir, '*.jpg'))

    # Limit the number of images
    image_paths = image_paths[:load_limit]

    # Create a TensorFlow dataset using tf.data
    image_paths_dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # Use tqdm to show progress while loading and preprocessing images
    image_dataset = image_paths_dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    # Convert the dataset to a numpy array
    dataset = np.array(list(tqdm(image_dataset.as_numpy_iterator(), total=len(image_paths))))
    
    print("====================================================================================================")
    print("Dataset shape")
    print("====================================================================================================")
    # Print dataset shape
    print(dataset.shape)

    return dataset