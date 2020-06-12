import argparse

from tensorflow.keras import losses, optimizers

import utils.constants as c
import utils.loader as l
from core.classifier import Classifier


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Trains a network with a customized architecture and data.')

    parser.add_argument(
        'dataset', help='Identifier to the dataset, without its absolute path', type=str)

    parser.add_argument(
        'output_model', help='Identifier to the output model, without its absolute path', type=str)

    parser.add_argument(
        '-height', help='Height of the images to be trained', type=int, default=100)

    parser.add_argument(
        '-width', help='Width of the images to be trained', type=int, default=100)

    parser.add_argument(
        '-n_classes', help='Number of classes to be trained', type=int, default=28)

    parser.add_argument(
        '-lr', help='Learning rate', type=float, default=1e-3)

    parser.add_argument(
        '-batch_size', help='Batch size', type=int, default=8)

    parser.add_argument(
        '-epochs', help='Number of training epochs', type=int, default=10)

    parser.add_argument(
        '-compress', help='Whether model file should be compressed after training', type=bool, default=False)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    output_model = args.output_model
    height = args.height
    width = args.width
    n_classes = args.n_classes
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    compress = args.compress

    # Creating data and model paths
    data_path = f'{c.DATA_FOLDER}/{dataset}/'
    model_path = f'{c.MODEL_FOLDER}/{output_model}'

    # Creating training and validation data generators
    train = l.create_generator(data_path + 'train', height, width, batch_size)
    val = l.create_generator(data_path + 'val', height, width, batch_size)

    # Instantiates a classifier
    clf = Classifier.new(height, width, n_classes)

    # Creating a optimizer and a loss function
    optimizer = optimizers.Adam(learning_rate=lr)
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)

    # Attaching properties to the classifier
    clf.compile(optimizer, loss)

    # Fitting the model with training data (validation data for early stopping)
    clf.fit(train, val, epochs=epochs)

    # Saving model
    clf.save(model_path, compress=compress)
