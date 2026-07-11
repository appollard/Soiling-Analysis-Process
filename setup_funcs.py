# bmp file type to uint8 greyscale image
import cv2
import numpy as np
from pathlib import Path


def file_to_img(file, background_colour):

    img_path = Path(__file__).parent / "images" / file
    img = cv2.imread(str(img_path))
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    if background_colour == "black":
        gray_img = 255 - gray_img
    elif background_colour == "white":
        pass
    else:
        raise ValueError("Acceptable values are 'black' or white'. Case sensitive.")
    return gray_img


def estimate_background_noise(img, block_size=10, lowest_n=5):
    """
    Break image into blocks, take the std and mean of the middle-quietest ones as the
    background noise estimate. Avoids particle regions without needing a particle
    mask.
    """

    # Exclusion threshold
    brightness_threshold = np.percentile(img, 75)

    h, w = img.shape
    blocks = []

    for r in range(0, h - block_size, block_size):
        for c in range(0, w - block_size, block_size):
            block = img[r : r + block_size, c : c + block_size]

            # Only consider blocks that are bright/dark enough to be background. No
            # need to worry about black/white background- that's already been
            # normalized
            if block.mean() < brightness_threshold:
                continue

            blocks.append((block.mean(), block.std()))

    blocks.sort(key=lambda x: x[1])  # sort by sd
    mid = len(blocks) // 2
    quietest = blocks[max(0, mid - lowest_n // 2) : mid + lowest_n // 2]

    mean = np.mean([p[0] for p in quietest])
    sd = np.mean([p[1] for p in quietest])

    return mean, sd
