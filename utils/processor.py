import cv2
import numpy as np

import utils.dictionary as d


def create_binary_mask(frame, kernel=3, lower_bound=[108, 23, 82], upper_bound=[179, 255, 255], dilate=False):
    """Creates a binary mask over the frame using image processing techniques.

    Args:
        frame (np.array): An array corresponding to the frame to be detected.
        kernel (int): Size of gaussian blur's kernel.
        lower_bound (list): Lower bound of HSV values.
        upper_bound (list): Upper bound of HSV values.
        dilate (bool): Whether mask should be dilated or not.

    Returns:
        The masked frame.

    """

    # Passes a gaussian blur over the frame
    blurred_frame = cv2.GaussianBlur(frame, (kernel, kernel), 0)

    # Converts the blurred frame into an HSV color format
    hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2HSV)

    # Creates the mask between lower and upper bounds possible HSV values
    masked_frame = cv2.inRange(hsv_frame, np.array(lower_bound), np.array(upper_bound))

    # Checks if it is supposed to apply dilation
    if dilate:
        # Passes a median blur over the mask
        blurred_frame = cv2.medianBlur(masked_frame, 5)

        # Creates a morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))

        # Dilates the masked frame
        masked_frame = cv2.dilate(blurred_frame, kernel, iterations=3)

        # Erodes the masked frame
        masked_frame = cv2.erode(masked_frame, kernel, iterations=3)

    return masked_frame


def create_gray_mask(frame, kernel=3):
    """Creates a grayscale mask over the frame using image processing techniques.

    Args:
        frame (np.array): An array corresponding to the frame to be detected.
        kernel (int): Size of gaussian blur's kernel.

    Returns:
        The masked frame.

    """

    # Passes a gaussian blur over the frame
    blurred_frame = cv2.GaussianBlur(frame, (kernel, kernel), 0)

    # Converts the blurred frame into an Grayscale color format
    masked_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2GRAY)

    return masked_frame


def detect_boxes(scores, boxes, height, width, threshold=0.75):
    """Selects predicted boxes that are over a certain threshold.

    Args:
        scores (tf.Tensor): A tensor containing the score of the detected boxes.
        boxes (tf.Tensor): A tensor containing the detected boxes.
        height (int): Height of the frame.
        width (int): Width of the frame.
        threshold (float): Threshold of detection score.

    Returns:
        A list of detected boxes among their positions on the frame.

    """

    # Creates a list of detected boxes
    detected_boxes = []

    # Iterates over every possible box
    for score, box in zip(scores[0], boxes[0]):
        # If score is bigger than threshold
        if score > threshold:
            # Gathers its box `left` and `right` positions
            left, right = int(box[1] * width), int(box[3] * width)

            # Gathers its box `top` and `bottom` positions
            top, bottom = int(box[0] * height), int(box[2] * height)

            # Appends as a tuple of positions to the list
            detected_boxes.append((left, right, top, bottom, score))

    return detected_boxes


def draw_boxes(frame, boxes, color=(77, 255, 9)):
    """Draws detection boxes in the frame.

    Args:
        frame (np.array): An array corresponding to the frame to be detected.
        boxes (list): List of tuples containing the boxes positioning.
        color (tuple): Tuple containing the color of the drawing box.

    """

    # Iterates over every possible box
    for box in boxes:
        # Gathers its position and its score
        left, right, top, bottom, score = box

        # Creates a text variable with the score in percentage
        text = f'{(score.numpy() * 100):.2f}%'

        # Puts the text on the frame
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw a rectangle on the frame, which indicates the box positioning
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3, 1)


def draw_label(mask, label, prob, color=(77, 255, 9)):
    """Draws the predicted label and its probability in the mask.

    Args:
        mask (np.array): An array corresponding to the mask to be classified.
        label (tf.Tensor): A tensor holding the predicted label.
        prob (tf.Tensor): A tensor holding the predicted probability.
        color (tuple): Tuple containing the color of the drawing box.

    """

    # Creates a text variable with the score in percentage
    text = f'Sinal: {d.LIBRAS[label.numpy()]} ({(prob.numpy() * 100):.2f}%)'

    # Puts the text on the frame
    cv2.putText(mask, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def pad_box(box, height, width, padding=10):
    """Pads a box with extra height and width for gathering an increased-size detection.

    Args:
        box (tuple): Tuple containing a single box positioning.
        height (int): Height of the frame.
        width (int): Width of the frame.
        padding (int): Amount of pixels to be padded on every corner.

    Returns:
        The padded box new positions.

    """

    # Gathers the box positioning
    left, right, top, bottom, _ = box

    # Subtracts the padding from the `left` corner
    left -= padding

    # If `left` gets to a negative number
    if left < 0:
        # Resets as zero
        left = 0

    # Sums the padding to the `right` corner
    right += padding

    # If `right` exceeds the width of the frame
    if right > width:
        # Resets as width
        right = width

    # Subtracts the padding from the `top` corner
    top -= padding

    # If `top` gets to a negative number
    if top < 0:
        # Resets as zero
        top = 0

    # Sums the padding to the `bottom` corner
    bottom += padding

    # If `bottom` exceeds the width of the frame
    if bottom > height:
        # Resets as width
        bottom = height

    return left, right, top, bottom
