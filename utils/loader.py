import pathlib

import tensorflow as tf


def load_from_web(file_name, url='http://download.tensorflow.org/models/object_detection'):
    """Loads an pre-trained model from any website.

    Args:
        file_name (str): Name of the model's file.
        url (str): Base URL to load the model.

    Returns:
        The model object itself.

    """

    # Defining the model's name
    model_name = f'{file_name}.tar.gz'

    # Defining the model's directory
    model_path = tf.keras.utils.get_file(file_name, f'{url}/{model_name}', untar=True, cache_subdir='', cache_dir='models')

    # Appending the path to `saved_model` folder
    model_path = f'{pathlib.Path(model_path)}/saved_model'

    # Loading the model
    model = tf.saved_model.load(str(model_path))

    # Defining as a serving model
    model = model.signatures['serving_default']

    return model
