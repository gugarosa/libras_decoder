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
