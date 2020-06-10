import pathlib

import tensorflow as tf

import utils.compressor as c
from classifiers.small import SmallCNN


class Classifier:
    """
    """

    def __init__(self):
        """
        """

        #
        self.model = None

    def __call__(self, x):
        """
        """

        return self.model(x)

    @classmethod
    def new(cls, height, width, n_channels, n_classes, model='small'):
        """
        """

        #
        clf = cls()

        #
        if model == 'small':
            #
            clf.model = SmallCNN(height, width, n_channels, n_classes)

        return clf

    @classmethod
    def load(cls, model_path):
        """
        """

        # Checking if path does not exists
        if not pathlib.Path(model_path).is_dir():
            # Extracts the compressed file
            c.untar_file(model_path)

        #
        clf = cls()

        #
        clf.model = tf.keras.models.load_model(model_path)

        return clf

    def compile(self, optimizer, loss, metrics=['accuracy']):
        """
        """

        #
        self.model.compile(optimizer, loss, metrics)

    def fit(self, train, val, epochs=10):
        """
        """

        #
        self.model.fit(train, epochs=epochs, validation_data=val)

    def evaluate(self, test):
        """
        """

        #
        self.model.evaluate(test)

    def save(self, model_path, compress=False):
        """
        """

        #
        tf.keras.models.save_model(self.model, model_path, save_format='tf')

        #
        if compress:
            #
            c.tar_file(model_path)
