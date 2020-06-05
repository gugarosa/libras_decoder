import argparse

import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from core.architectures.small import SmallCNN


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(
        usage='Trains a network with custom data.')

    parser.add_argument(
        '-dataset', help='Identifier to the dataset', type=str, default='libras')

    parser.add_argument(
        '-height', help='Height of the images', type=int, default=320)

    parser.add_argument(
        '-width', help='Width of the images', type=int, default=240)

    parser.add_argument(
        '-n_channels', help='Number of channels of the images', type=int, default=3)

    parser.add_argument(
        '-lr', help='Learning rate', type=float, default=1e-3)

    parser.add_argument(
        '-batch_size', help='Batch size', type=int, default=1)

    parser.add_argument(
        '-epochs', help='Number of training epochs', type=int, default=100)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    height = args.height
    width = args.width
    n_channels = args.n_channels
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    # Creating training and validation data generators
    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)
    test_gen = ImageDataGenerator(rescale=1./255)

    #
    train = train_gen.flow_from_directory(batch_size=batch_size, directory=f'data/{dataset}/train',
                                       shuffle=True, target_size=(height, width),
                                       class_mode='binary')

    #
    val = val_gen.flow_from_directory(batch_size=batch_size, directory=f'data/{dataset}/val',
                                      shuffle=True, target_size=(height, width),
                                      class_mode='binary')

    #
    test = test_gen.flow_from_directory(batch_size=batch_size, directory=f'data/{dataset}/test',
                                        shuffle=True, target_size=(height, width),
                                        class_mode='binary')

    #
    model = SmallCNN(height=height, width=width, n_channels=n_channels)

    #
    optimizer = optimizers.Adam(learning_rate=lr)

    #
    model.compile(optimizer, loss=losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    #
    model.fit_generator(train, steps_per_epoch=batch_size, epochs=epochs, validation_data=val)

    #
    model.evaluate(val)
