import argparse
import pathlib

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import utils.constants as c
import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Evaluates a pre-trained network.')

    parser.add_argument(
        'dataset', help='Identifier to the dataset', type=str, default='libras')

    parser.add_argument(
        '-height', help='Height of the images', type=int, default=320)

    parser.add_argument(
        '-width', help='Width of the images', type=int, default=240)

    parser.add_argument(
        '-batch_size', help='Batch size', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    height = args.height
    width = args.width
    batch_size = args.batch_size

    # Creating path to the model itself
    model_path = f'{c.MODEL_FOLDER}/{dataset}'

    # Creating training and validation data generators
    test_gen = ImageDataGenerator(rescale=1./255)

    # Creating a test data generator
    test = test_gen.flow_from_directory(batch_size=batch_size, directory=f'{c.DATA_FOLDER}/{dataset}/test',
                                        shuffle=True, target_size=(height, width),
                                        class_mode='binary')

    # Checking if path does not exists
    if not pathlib.Path(model_path).is_dir():
        # Extracts the compressed file
        l.untar_file(model_path)

    # Loading model
    model = tf.keras.models.load_model(model_path)

    # Evaluates the model on test set
    model.evaluate(test)
