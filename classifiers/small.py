import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)


class SmallCNN(tf.keras.Model):
    """A SmallCNN class implements a standard CNN architecture.

    """

    def __init__(self, height, width, n_channels, n_classes=1):
        """Initialization method.

        Args:
            height (int): Height of input image.
            width (int): Width of input image.
            n_channels (int): Number of channels from input image.
            n_classes (int): Number of classes.

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
        self.fc2 = Dense(n_classes)

    def call(self, x):
        """Performs a forward pass over the model.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
            The output of the network.

        """

        # First convolutional + pooling block
        x = self.pool1(self.conv1(x))

        # Second convolutional + pooling block
        x = self.pool2(self.conv2(x))

        # Third convolutional + pooling block
        x = self.pool3(self.conv3(x))

        # Flattens the output
        x = self.flatten(x)

        # First fully-connected layer
        x = self.fc1(x)

        # Output layer
        x = self.fc2(x)

        return x
