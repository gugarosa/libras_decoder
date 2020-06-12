import pathlib

import tensorflow as tf

import utils.compressor as c
from classifiers.small import SmallCNN


class Classifier:
    """A Classifier class abstracts the Tensorflow API and provides a more clearer usage.

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

        # Gathering height and width from model
        height, width = self.model.inputs[0].shape[1], self.model.inputs[0].shape[2]

        #
        x = tf.expand_dims(tf.expand_dims(x, -1), 0)

        # Resizing the input
        x = tf.image.resize(x, [height, width])

        #
        x = tf.nn.softmax(self.model(x / 255))

        #
        label = tf.argmax(x, axis=1)

        #
        prob = tf.gather(x, label, axis=1)

        return label, prob

    @classmethod
    def new(cls, height, width, n_classes, model='small'):
        """Creates a new classifier using a pre-defined architecture.

        Args:
            height (int): Height of the images.
            width (int): Width of the images.
            n_classes (int): Number of classes.

        Returns:
            A Classifier object.

        """

        # Instantiates the class
        clf = cls()

        # If it is supposed to use the `small` architecture
        if model == 'small':
            # Instantiates the architecture
            clf.model = SmallCNN(height, width, n_classes)

        return clf

    @classmethod
    def load(cls, model_path):
        """Loads a classifier from a pre-trained model.

        Args:
            model_path (str): Path to the pre-trained model.

        Returns:
            A Classifier object.

        """

        # Checking if path does not exists
        if not pathlib.Path(model_path).is_dir():
            # Extracts the compressed file
            c.untar_file(model_path)

        # Instantiates the class
        clf = cls()

        # Loads the pre-trained model
        clf.model = tf.keras.models.load_model(model_path)

        return clf

    def compile(self, optimizer, loss, metrics=['accuracy']):
        """Compiles the model.

        Args:
            optimizer (tf.keras.optimizers): An optimizer object.
            loss (tf.keras.losses): A loss object.
            metrics (list): A list of metrics to be used.

        """

        # Compiles the model
        self.model.compile(optimizer, loss, metrics)

    def fit(self, train, val, epochs=10):
        """Fits the model.

        Args:
            train (tf.keras.preprocessing.image.ImageDataGenerator): Training data.
            val (tf.keras.preprocessing.image.ImageDataGenerator): Validation data.
            epochs (int): Number of training epochs.

        """

        # Fits the model
        self.model.fit(train, epochs=epochs, validation_data=val)

    def evaluate(self, test):
        """Evaluates the model.

        Args:
            test (tf.keras.preprocessing.image.ImageDataGenerator): Testing data.

        """

        # Evaluates the model
        self.model.evaluate(test)

    def save(self, model_path, compress=False):
        """Saves the model to a folder.

        Args:
            model_path (str): Path to saved the model.
            compress (bool): Whether folder should be compressed or not.

        """

        # Saves the model using TF
        tf.keras.models.save_model(self.model, model_path, save_format='tf')

        # If it is supposed to be compressed
        if compress:
            # Compress the folder using .tar.gz
            c.tar_file(model_path)
