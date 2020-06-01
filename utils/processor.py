import cv2
import numpy as np


def create_mask(frame, kernel=3, lower_bound=[108, 23, 82], upper_bound=[179, 255, 255], dilate=False):
    """Creates a mask over the frame using image processing techniques.

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
    masked_frame = cv2.inRange(hsv_frame, np.array(
        lower_bound), np.array(upper_bound))

    # Checks if it is supposed to apply dilation
    if dilate:
        # Passes a median blur over the mask
        m_blurred_frame = cv2.medianBlur(masked_frame, 5)

        # Creates a dilating kernel
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

        # Dilates the masked frame
        masked_frame = cv2.dilate(m_blurred_frame, dilate_kernel)

    return masked_frame
