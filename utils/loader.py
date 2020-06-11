import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import utils.constants as c


def create_generator(data_path, height, width, batch_size=1,
                     color='grayscale', label='sparse',
                     shuffle=True, rescale=True):
    """Creates a generator using Tensorflow toolkit.

    Args:
        data_path (str): Path to the directory of data.
        height (int): Height of the images.
        width (int): Width of the images.
        batch_size (int): Number of images per batch.
        color (str): Coloring of the images, e.g., `grayscale`, `rgb`, `rgba`.
        label (str): Labels identifier, e.g., `binary`, `categorical`, `sparse`.
        shuffle (bool): Whether data should be shuffled or not.
        rescale (bool): Whether data should be rescaled or not.

    Returns:
        The generator itself.

    """

    # Checks if it is supposed to rescale
    if rescale:
        # Creates the generator
        generator = ImageDataGenerator(rescale=1./255)

    # If it is not
    else:
        # Creates the generator without rescaling
        generator = ImageDataGenerator()

    # Gathers the data itself from the generator
    data = generator.flow_from_directory(directory=data_path, target_size=(height, width),
                                         batch_size=batch_size, color_mode=color,
                                         class_mode=label, shuffle=shuffle)

    return data


def download_model(base_url, file_name):
    """Downloada a pre-trained model from the internet.

    Args:
        base_url (str): Base URL of the model to be downloaded.
        file_name (str): Name of the file to be downloaded.

    """

    # Creates the URL to be downloaded
    url = f'{base_url}/{file_name}'

    # Downloads the file
    tf.keras.utils.get_file(file_name, url, cache_subdir='', cache_dir=c.MODEL_FOLDER)
