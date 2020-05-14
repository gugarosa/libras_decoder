import pathlib

import tensorflow as tf

# Defines a constant for the base URL to load the model
BASE_URL = 'http://download.tensorflow.org/models/object_detection'


def load_from_zoo(file_name):
    """Loads an pre-trained model from tensorflow's zoo.

    Args:
        file_name (str): Name of the model's file.

    Returns:
        The model object itself.

    """

    # Defining the model's name
    model_name = f'{file_name}.tar.gz'

    # Defining the model's directory
    model_dir = tf.keras.utils.get_file(fname=file_name, origin=f'{BASE_URL}/{model_name}', untar=True)

    # Appending the path to `saved_model` folder
    model_dir = f'{pathlib.Path(model_dir)}/saved_model'

    # Loading the model
    model = tf.saved_model.load(str(model_dir))

    # Defining as a serving model
    model = model.signatures['serving_default']

    return model
