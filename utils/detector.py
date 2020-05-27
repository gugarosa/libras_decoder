import cv2
import tensorflow as tf


def detect_frame(model, frame):
    """Performs an inference over the model and detects the objects' bounding boxes.

    Args:
        model (tf.Model): A pre-trained Tensorflow model.
        frame (np.array): An array corresponding to the frame to be detected.

    Returns:
        A dictionary containing the predictions (labels and bounding boxes).

    """

    # Converting the frame to a tensor
    tensor = tf.convert_to_tensor(frame)

    # Expanding the first axis, i.e., `batch size`
    tensor = tf.expand_dims(tensor, axis=0)

    # Running the prediction over the model
    preds = model(tensor)

    return preds


def detect_box(frame, scores, boxes, height, width, threshold=0.75):
    """
    """

    #
    detected_boxes = []

    #
    scores, boxes = tf.squeeze(scores), tf.squeeze(boxes)

    #
    for score, box in zip(scores, boxes):
        #
        if score > threshold:
            #
            left, right = int(box[1] * width), int(box[3] * width)

            #
            top, bottom = int(box[0] * height), int(box[2] * height)

            #
            detected_boxes.append((left, right, top, bottom, score))

    return detected_boxes


def draw_box(frame, boxes, color=(77, 255, 9)):
    """
    """

    #
    for box in boxes:
        #
        left, right, top, bottom, score = box

        #
        text = f'{(score.numpy() * 100):.2f}%'

        #
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        #
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3, 1)


def pad_box(box, height, width, padding=10):
    """
    """

    left, right, top, bottom, _ = box

    #
    left -= padding

    if left < 0:
        left = 0

    #
    right += padding
    
    if right > width:
        right = width

    #
    top -= padding

    if top < 0:
        top = 0

    #
    bottom += padding

    if bottom > height:
        bottom = height

    return left, right, top, bottom