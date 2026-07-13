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


def save_dog_product(
    img, filename, folder_name="Output Files/DoG stage images", base_dir=None
):
    """Save an image to a named folder next to the script."""

    if base_dir is None:
        base_dir = os.path.dirname(__file__)

    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    # Handle boolean arrays
    if np.asarray(img).dtype == bool:
        img = (np.asarray(img) * 255).astype(np.uint8)
    cv2.imwrite(output_path, img)
    return


def full_dog_stage(
    img, min_radius_big_particles, um_per_pixel, min_radius_small_particles=0
):
    """
    Identify the best pair of sigma values for the DoG (WIP). .....
    """

    # Too harsh rn
    # if check_suitability(img):
    #    return (None, img, None)

    # WIP####
    # best_sigmas, responses = find_optimal_dog_sigmas(
    #    img, r_min=1, r_max=min_radius_big_particles * 5, n_scales=30
    # )
    # plot_scale_responses(responses)
    #########
    s1 = 1
    s2 = 2

    mask, _ = apply_dog_triangle(img, s1, s2)
    filled_mask = fill_outlines(mask)
    particle_dicts = identify_small_particles(
        filled_mask, min_radius_small_particles, um_per_pixel
    )
    new_img, av_grey = remove_particles(img, particle_dicts)
    save_dog_product(255 - (255 * mask), "DoG Mask.png")
    save_dog_product(new_img, "Small particles removed.png")

    return particle_dicts, new_img, filled_mask
