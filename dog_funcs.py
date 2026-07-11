import numpy as np
import os
import cv2
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops


# If image is nearly uniform, nothing left for DoG to find
def check_suitability(img):
    """
    If all particles are a similar intensity (like in the test images) just skip
    the DoG entirely.
    """

    if img.std() < 5:
        print("Image too uniform for DoG — skipping")
        return True
    else:
        return False


def find_optimal_dog_sigmas(img, r_min=2, r_max=30, n_scales=20):
    """
    Sweep sigma pairs across a range of particle radii and return
    the one with the strongest normalised DoG response.
    """
    sigma_opts = np.linspace(r_min / np.sqrt(2), r_max / np.sqrt(2), n_scales)

    best_response = -np.inf
    best_sigmas = None
    responses = []

    for s_opt in sigma_opts:
        s1 = s_opt / np.sqrt(1.6)
        s2 = s_opt * np.sqrt(1.6)

        dog = gaussian_filter(img.astype(float), s1) - gaussian_filter(
            img.astype(float), s2
        )

        # Normalise by sigma^2 so scales are comparable
        normalised = np.max(np.abs(dog)) * s_opt**2
        responses.append((s_opt, s1, s2, normalised))

        if normalised > best_response:
            best_response = normalised
            best_sigmas = (s1, s2)

    return best_sigmas, responses


def plot_scale_responses(responses):
    """
    Plot response strength vs scale so you can see the full distribution,
    not just the single best.
    """
    import matplotlib.pyplot as plt

    sigma_opts = [r[0] for r in responses]
    strengths = [r[3] for r in responses]
    plt.plot(sigma_opts, strengths)
    plt.xlabel("sigma_optimal (px)")
    plt.ylabel("normalised DoG response")
    plt.title("DoG response vs scale")
    plt.show()

    return


def apply_dog_triangle(img, s1, s2):
    """
    Apply DoG with triangle thresholding to identify small particles.
    """
    img_float = img.astype(float)
    dog = gaussian_filter(img_float, s1) - gaussian_filter(img_float, s2)

    thresh = threshold_triangle(dog)
    mask = dog < thresh

    return mask, dog


def fill_outlines(mask):
    """
    Fill the particle outlines produced by the DoG
    """

    mask_filled = binary_fill_holes(mask)

    return mask_filled


def identify_small_particles(mask, min_radius, um_per_pixel):
    """
    Identify the small particles in the mask produced by apply_otsu. Returns a list of
    dicts- one for each particle. Remove any particles with a radius (in pixels) below
    a certain point.
    """

    # Label connected regions (each particle gets a unique integer)
    labelled = label(mask)
    props = regionprops(labelled, intensity_image=mask)

    particle_dicts = []
    for p in props:
        # coords from regionprops are (row, col) tuples, same as original
        coords = [tuple(c) for c in p.coords.tolist()]

        particle_dicts.append(
            {
                "coords": coords,
                "centroid": (
                    p.centroid[1] * um_per_pixel,  # x in um
                    p.centroid[0] * um_per_pixel,
                ),  # y in um
                "effective_diameter": p.equivalent_diameter_area * um_per_pixel,
                "major_axis": p.axis_major_length * um_per_pixel,
                "minor_axis": p.axis_minor_length * um_per_pixel,
                "orientation": p.orientation,
                "pixel_count": len(coords),
                "area": p.area * (um_per_pixel**2),
                # kept as None — no ellipse method being run
                "outline_coords": None,
                "circumference": None,
                "corrected_diameter": None,
                "corrected_area": None,
            }
        )

    particle_dicts = [
        p for p in particle_dicts if np.sqrt(p["pixel_count"] / np.pi) >= min_radius
    ]

    return particle_dicts


def remove_particles(img, particle_dicts):
    """
    Take the original microscope image and replace all pixels which contain identified particles
    with the average grey of the image. This is crude and assumes low particle density.
    """

    new_img = img.copy()
    av_grey = np.mean(img)
    for p in particle_dicts:
        for row, col in p["coords"]:
            new_img[row, col] = av_grey
    return new_img, av_grey


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
