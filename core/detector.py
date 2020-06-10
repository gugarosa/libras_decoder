import pathlib

import tensorflow as tf

import utils.compressor as c


class Detector:
    """A Detector class abstracts the usage of pre-trained models from Tensorflow Object Detection API.

    """

    def __init__(self):
        """Initialization method.

        """

        # There is no model in its default initialization
        self.model = None

    def __call__(self, x):
        """Returns a specific output whenever this class is called.

        Returns:
            Returns the prediction over the model itself.

        """

        return self.model(x)

    @classmethod
    def load(cls, model_path):
        """Loads an object detection pre-trained model.

        Args:
            model_path (str): Path to the pre-trained model.

        Returns:
            A Detector object.

        """

        # Checking if path does not exists
        if not pathlib.Path(model_path).is_dir():
            # Extracts the compressed file
            c.untar_file(model_path)

        # Appending `saved_model` folder to the path
        appended_path = f'{model_path}/saved_model'

        # Instantiates the class
        clf = cls()

        # Loads the pre-trained model
        clf.model = tf.saved_model.load(appended_path)

        # Defining as a serving model
        clf.model = clf.model.signatures['serving_default']

        return clf
