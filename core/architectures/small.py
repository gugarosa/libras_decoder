import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)


class SmallCNN(tf.keras.Model):
    """
    """

    def __init__(self, height=320, width=240, n_channels=3):
        """Initialization method.

        Args:
            height (int):
            width (int):
            n_channels (int):

        """

        # Overriding class with custom properties
        super(SmallCNN, self).__init__(name='small_cnn')

        # Defining convolutional layers
        self.conv1 = Conv2D(16, 3, padding='same', activation='relu')
        self.conv2 = Conv2D(32, 3, padding='same', activation='relu')
        self.conv3 = Conv2D(64, 3, padding='same', activation='relu')

        # Defining pooling layers
        self.pool1 = MaxPooling2D()
        self.pool2 = MaxPooling2D()
        self.pool3 = MaxPooling2D()

        # Defining the flattenning layer
        self.flatten = Flatten()

        # Defining the fully-connected layers
        self.fc1 = Dense(512, activation='relu')
        self.fc2 = Dense(1)

    def call(self, x):
        """Performs a forward pass over the model.

        Args:
            x (tf.Tensor):

        Returns:


        """

        #
        x = self.pool1(self.conv1(x))

        #
        x = self.pool2(self.conv2(x))

        #
        x = self.pool3(self.conv3(x))

        #
        x = self.flatten(x)

        #
        x = self.fc1(x)

        #
        x = self.fc2(x)

        return x
