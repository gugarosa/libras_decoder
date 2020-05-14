import argparse

import cv2
import imutils

import utils.detector as d
import utils.loader as l
from utils.video_stream import VideoStream


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Streams a video from an input device.')

    parser.add_argument(
        '-model', help='Identifier to the input pre-trained model', type=str, default='ssd_mobilenet_v1_coco_2017_11_17')

    parser.add_argument(
        '-device', help='Identifier to the input streaming device', type=int, default=0)

    parser.add_argument(
        '-width', help='Width of the captured frames', type=int, default=480)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    model_name = args.model
    device = args.device
    width = args.width

    # Starts a thread from the `VideoStream` class
    v = VideoStream(device).start_thread()

    # Loading the model from Tensorflow's Zoo
    model = l.load_from_zoo(model_name)

    # While the loop is True
    while True:
        # Reads a new frame
        frame = v.read()

        # Resizes the frame
        frame = imutils.resize(frame, width=width)

        # Performing the detection over the frame
        preds = d.detect_frame(model, frame)

        print(preds)

        # Shows the frame using `open-cv`
        cv2.imshow('frame', frame)

        # If the `q` key is inputted, breaks the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy all windows for cleaning up memory
    cv2.destroyAllWindows()

    # Stop the thread
    v.stop_thread()
