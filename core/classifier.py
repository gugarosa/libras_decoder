import tensorflow as tf

import utils.loader as l
from classifiers.small import SmallCNN


class Classifier:
    """
    """

    def __init__(self, height, width, n_channels, n_classes, model='small'):
        """
        """
        
        #
        if model == 'small':
            #
            self.model = SmallCNN(height, width, n_channels, n_classes)

    def __call__(self, x):
        """
        """

        return self.model(x)

    @classmethod
    def load(cls, model_path):
        """
        """

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

    def fit(self, train, val, batch_size=2, epochs=10):
        """
        """

        #
        self.model.fit(train, epochs=epochs, validation_data=val)

    def save(self, model_path, compress=False):
        """
        """

        #
        tf.keras.models.save_model(self.model, model_path, save_format='tf')

        #
        if compress:
            #
            l.tar_file(model_path)
