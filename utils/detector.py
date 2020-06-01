import cv2
import tensorflow as tf


def predict_over_frame(model, frame):
    """Performs an inference over a model.

    Args:
        model (tf.Model): A pre-trained Tensorflow model.
        frame (np.array): An array corresponding to the frame to be detected.

    Returns:
        An object, commonly a dictionary, containing the predictions.

    """

    # Converting the frame to a tensor
    tensor = tf.convert_to_tensor(frame)

    # Expanding the first axis, i.e., `batch size`
    tensor = tf.expand_dims(tensor, axis=0)

    # Running the prediction over the model
    preds = model(tensor)

    return preds


def detect_boxes(frame, scores, boxes, height, width, threshold=0.75):
    """Selects predicted boxes that are over a certain threshold.

    Args:
        frame (np.array): An array corresponding to the frame to be detected.
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

    # Removes the batch dimension from scores and boxes
    scores, boxes = tf.squeeze(scores), tf.squeeze(boxes)

    # Iterates over every possible box
    for score, box in zip(scores, boxes):
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
        cv2.putText(frame, text, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Draw a rectangle on the frame, which indicates the box positioning
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3, 1)


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
