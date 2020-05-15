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


def draw_boxes(frame, scores, boxes, labels, height, width, threshold=0.75, color=(77, 255, 9)):
    """
    """

    #
    scores, boxes, labels = tf.squeeze(scores), tf.squeeze(boxes), tf.squeeze(labels)

    #
    for score, box, label in zip(scores, boxes, labels):
        #
        if score > threshold:
            #
            left, right = int(box[1] * width), int(box[3] * width)

            #
            top, bottom = int(box[0] * height), int(box[2] * height)

            #
            text = f'{label}: {(score.numpy() * 100):.2f}%'

            #
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            #
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3, 1)
