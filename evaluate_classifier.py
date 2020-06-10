import argparse

import utils.constants as c
import utils.loader as l
from core.classifier import Classifier


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Evaluates a pre-trained network.')

    parser.add_argument(
        'dataset', help='Identifier to the dataset, without its absolute path', type=str)

    parser.add_argument(
        'input_model', help='Identifier to the pre-trained model, without its absolute path', type=str)

    parser.add_argument(
        '-height', help='Height of the images to be evaluated', type=int, default=100)

    parser.add_argument(
        '-width', help='Width of the images to be evaluated', type=int, default=100)

    parser.add_argument(
        '-batch_size', help='Batch size', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    input_model = args.input_model
    height = args.height
    width = args.width
    batch_size = args.batch_size

    # Creating data and model paths
    data_path = f'{c.DATA_FOLDER}/{dataset}/'
    model_path = f'{c.MODEL_FOLDER}/{input_model}'

    # Creating testing data generator
    test = l.create_generator(data_path + 'test', height, width, batch_size)

    # Instantiates a classifier from pre-trained model
    clf = Classifier.load(model_path)

    # Evaluates the model on test set
    clf.evaluate(test)
