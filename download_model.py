import argparse

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(
        usage='Loads a pre-trained model from the web.')

    parser.add_argument(
        '-model', help='Identifier to the input pre-trained model name', type=str, default='ssd_mobilenet_v1_coco_2017_11_17')

    parser.add_argument(
        '-url', help='Identifier to the input pre-trained model URL', type=str, default='http://download.tensorflow.org/models/object_detection')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    model_name = args.model
    model_url = args.url

    # Loading the model from web
    model = l.load_from_web(model_name, model_url)
