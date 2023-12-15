import cv2
import numpy as np
from skimage import transform as trans

# Define a standard set of destination landmarks for ArcFace alignment
arcface_dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(lmk, image_size=112, mode="arcface"):
    """
    Estimate the transformation matrix for aligning facial landmarks.

    Args:
        lmk (numpy.ndarray): 2D array of shape (5, 2) representing facial landmarks.
        image_size (int): Desired output image size.
        mode (str): Alignment mode, currently only "arcface" is supported.

    Returns:
        numpy.ndarray: Transformation matrix (2x3) for aligning facial landmarks.
    """
    # Check input conditions
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0

    # Adjust ratio and x-coordinate difference based on image size
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio

    # Scale and shift the destination landmarks
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x

    # Estimate the similarity transformation
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]

    return M


def norm_crop(img, landmark, image_size=112, mode="arcface"):
    """
    Normalize and crop a facial image based on provided landmarks.

    Args:
        img (numpy.ndarray): Input facial image.
        landmark (numpy.ndarray): 2D array of shape (5, 2) representing facial landmarks.
        image_size (int): Desired output image size.
        mode (str): Alignment mode, currently only "arcface" is supported.

    Returns:
        numpy.ndarray: Normalized and cropped facial image.
    """
    # Estimate the transformation matrix
    M = estimate_norm(landmark, image_size, mode)

    # Apply the affine transformation to the image
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

    return warped
