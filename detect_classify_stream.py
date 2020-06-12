import argparse

import cv2

import utils.processor as p
from core.classifier import Classifier
from core.detector import Detector
from core.stream import Stream


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Predicts information from a streamming video.')

    parser.add_argument(
        'detector', help='Identifier to the input pre-trained detection model', type=str)

    parser.add_argument(
        'classifier', help='Identifier to the input pre-trained classification model', type=str)

    parser.add_argument(
        '-device', help='Identifier to the input streaming device', type=int, default=0)

    parser.add_argument(
        '-height', help='Height of the captured frames', type=int, default=320)

    parser.add_argument(
        '-width', help='Width of the captured frames', type=int, default=480)

    parser.add_argument(
        '-threshold', help='Threshold to display the object detection', type=float, default=0.25)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    detector = args.detector
    classifier = args.classifier
    device = args.device
    height = args.height
    width = args.width
    threshold = args.threshold

    # Starts a thread from the `Stream` class
    v = Stream(height, width, device).start_thread()

    # Loading the detection model
    det = Detector.load(f'models/{detector}')

    # Loading the classification model
    clf = Classifier.load(f'models/{classifier}')

    # While the loop is True
    while True:
        # Reads a new frame
        valid, frame = v.read()

        # Checks if the frame is valid
        if valid:
            # Performing the detection over the frame
            det_preds = det(frame)

            # Detects bounding boxes over the objects
            detected_boxes = p.detect_boxes(det_preds['detection_scores'], det_preds['detection_boxes'],
                                            height, width, threshold=threshold)

            # If the amount os detected boxes is larger than zero
            if len(detected_boxes) > 0:
                # Gathers the box positions
                left, right, top, bottom = p.pad_box(detected_boxes[0], height, width, padding=50)

                # Defines the region of interest (ROI)
                roi = frame[top:bottom, left:right, :]

                # Shows the hand itself
                cv2.imshow(f'hand', roi)

                # Creates a mask using the ROI
                mask = p.create_mask(roi, dilate=False)

                # Performing the classification over the mask
                clf_label, clf_prob = clf(mask)

                # Converting the mask to a BGR (just for inputting the text)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # Draws the predicted label and its probability
                p.draw_label(mask, clf_label, clf_prob)

                # Shows the mask
                cv2.imshow(f'mask', mask)

            # Draw bounding boxes according to detected objects
            p.draw_boxes(frame, detected_boxes)

            # Shows the frame using `open-cv`
            cv2.imshow('stream', frame)

        # If the `q` key is inputted, breaks the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy all windows for cleaning up memory
    cv2.destroyAllWindows()

    # Stop the thread
    v.stop_thread()
