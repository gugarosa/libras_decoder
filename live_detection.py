import argparse

import cv2
import imutils

import utils.detector as d
import utils.loader as l
from utils.video_stream import VideoStream
import numpy as np


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Streams a video from an input device.')

    parser.add_argument(
        '-model', help='Identifier to the input pre-trained model', type=str, default='ssd_mobilenet_v2_egohands')

    parser.add_argument(
        '-url', help='Identifier to the input pre-trained model URL', type=str, default='http://recogna.tech/files')

    parser.add_argument(
        '-device', help='Identifier to the input streaming device', type=int, default=0)

    parser.add_argument(
        '-height', help='Height of the captured frames', type=int, default=270)

    parser.add_argument(
        '-width', help='Width of the captured frames', type=int, default=480)

    parser.add_argument(
        '-threshold', help='Threshold to display the object detection', type=float, default=0.25)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    model_name = args.model
    model_url = args.url
    device = args.device
    height = args.height
    width = args.width
    threshold = args.threshold

    # Starts a thread from the `VideoStream` class
    v = VideoStream(device).start_thread()

    # Loading the model from web
    model = l.load_from_web(model_name, model_url)

    # While the loop is True
    while True:
        # Reads a new frame
        valid, frame = v.read()

        #
        if valid:

            # Resizes the frame
            frame = imutils.resize(frame, height=height, width=width)

            # Flips the frame
            frame = cv2.flip(frame, 1)

            # Performing the detection over the frame
            preds = d.detect_frame(model, frame)

            # Detects bounding boxes over the objects
            detected_boxes = d.detect_box(frame, preds['detection_scores'], preds['detection_boxes'],
                                          height, width, threshold=threshold)

            # If the amount os detected boxes is larger than zero
            if len(detected_boxes) > 0:
                # Gathers the box positions
                left, right, top, bottom = d.pad_box(detected_boxes[0], height, width, padding=25)

                # Defines the region of interest
                roi = frame[top:bottom, left:right, :]

                # Shows the hand itself
                cv2.imshow(f'hand', roi)

                blur = cv2.GaussianBlur(roi, (3,3), 0)
                hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

                lower_color = np.array([108, 23, 82])
                upper_color = np.array([179, 255, 255])

                mask = cv2.inRange(hsv, lower_color, upper_color)
                blur = cv2.medianBlur(mask, 5)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
                mask2 = cv2.dilate(blur, kernel)

                cv2.imshow(f'mask', mask2)

            # Draw bounding boxes according to detected objects
            d.draw_box(frame, detected_boxes)

            # Shows the frame using `open-cv`
            cv2.imshow('stream', frame)

        # If the `q` key is inputted, breaks the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy all windows for cleaning up memory
    cv2.destroyAllWindows()

    # Stop the thread
    v.stop_thread()
