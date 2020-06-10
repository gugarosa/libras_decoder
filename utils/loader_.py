import pathlib
import tarfile

import tensorflow as tf

import utils.constants as c


def load_from_disk(model_name):
    """Loads an pre-trained model from disl.

    Args:
        model_name (str): Name of the model's.

    Returns:
        The model object itself.

    """

    # Appending the path to `saved_model` folder
    model_path = f'{c.MODEL_FOLDER}/{pathlib.Path(model_name)}/saved_model'

    # Loading the model
    model = tf.saved_model.load(model_path)

    # Defining as a serving model
    model = model.signatures['serving_default']

    return model


def load_from_web(model_name, url='http://download.tensorflow.org/models/object_detection'):
    """Loads an pre-trained model from any website.

    Args:
        model_name (str): Name of the model's.
        url (str): Base URL to load the model.

    Returns:
        The model object itself.

    """

    # Defining the model's name
    file_name = f'{model_name}.tar.gz'

    # Defining the model's directory
    model_path = tf.keras.utils.get_file(model_name, f'{url}/{file_name}', untar=True, cache_subdir='', cache_dir=c.MODEL_FOLDER)    

    # Appending the path to `saved_model` folder
    model_path = f'{pathlib.Path(model_path)}/saved_model'

    # Loading the model
    model = tf.saved_model.load(model_path)

    # Defining as a serving model
    model = model.signatures['serving_default']

    return model
