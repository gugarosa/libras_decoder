import argparse

import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import utils.constants as c
import utils.loader as l
from core.architectures.small import SmallCNN


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Trains a network with customized architecture and data.')

    parser.add_argument(
        'dataset', help='Identifier to the dataset', type=str, default='libras')

    parser.add_argument(
        '-height', help='Height of the images', type=int, default=100)

    parser.add_argument(
        '-width', help='Width of the images', type=int, default=100)

    parser.add_argument(
        '-n_channels', help='Number of channels of the images', type=int, default=1)

    parser.add_argument(
        '-n_classes', help='Number of classes', type=int, default=4)

    parser.add_argument(
        '-lr', help='Learning rate', type=float, default=1e-3)

    parser.add_argument(
        '-batch_size', help='Batch size', type=int, default=2)

    parser.add_argument(
        '-epochs', help='Number of training epochs', type=int, default=20)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    height = args.height
    width = args.width
    n_channels = args.n_channels
    n_classes = args.n_classes
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    # Creating training and validation data generators
    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)

    # Creating a train data generator
    train = train_gen.flow_from_directory(batch_size=batch_size, directory=f'{c.DATA_FOLDER}/{dataset}/train',
                                          color_mode='grayscale', shuffle=True, target_size=(height, width),
                                          class_mode='sparse')

    # Creating a validation data generator
    val = val_gen.flow_from_directory(batch_size=batch_size, directory=f'{c.DATA_FOLDER}/{dataset}/val',
                                      color_mode='grayscale', shuffle=True, target_size=(height, width),
                                      class_mode='sparse')

    # Creating the model itself
    model = SmallCNN(height=height, width=width, n_channels=n_channels, n_classes=n_classes)

    # Creating an optimizer
    optimizer = optimizers.Adam(learning_rate=lr)

    # Attaching optimizer, loss and metrics to the model
    model.compile(optimizer, loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # Fitting the model with training data (validation data for early stopping)
    model.fit_generator(train, steps_per_epoch=batch_size, epochs=epochs, validation_data=val)

    # Saving model
    tf.keras.models.save_model(model, f'{c.MODEL_FOLDER}/{dataset}', save_format='tf')

    # Compress the file into a .tar.gz
    # l.tar_file(dataset)
