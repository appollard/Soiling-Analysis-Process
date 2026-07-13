from skimage.filters import threshold_otsu


def apply_otsu(img):
    """
    Generate a mask for the image with a threshold determined by the otsu function.
    This minimizes the variation between the greyscale values of 'clean' and 'soiled'
    particles. This means that it ignores smaller particles.
    """

    thresh = threshold_otsu(img)
    mask = img < thresh

    return mask
