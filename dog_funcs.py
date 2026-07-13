import numpy as np
import os
import cv2
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops


def apply_dog_triangle(img, s1, s2):
    """
    Apply DoG with triangle thresholding to identify small particles.
    """
    img_float = img.astype(float)
    dog = gaussian_filter(img_float, s1) - gaussian_filter(img_float, s2)

    thresh = threshold_triangle(dog)
    mask = dog < thresh

    return mask, dog
