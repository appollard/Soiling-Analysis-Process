# Packages
import numpy as np
import os
import cv2
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops

# Files
import otsu_funcs as otsu
import dog_funcs as dog
import visualiser_funcs as visualiser
import setup_funcs as setup
import testing_funcs as testing


def apply_dog_triangle(img, s1, s2):
    """
    Apply DoG with triangle thresholding to identify small particles.
    """
    img_float = img.astype(float)
    dog = gaussian_filter(img_float, s1) - gaussian_filter(img_float, s2)

    thresh = threshold_triangle(dog)
    mask = dog < thresh

    return mask


def fill_outlines(mask):
    """
    Fill the particle outlines produced by the DoG
    """

    mask_filled = binary_fill_holes(mask)

    return mask_filled


def identify_small_particles(mask, um_per_pixel):
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

    return particle_dicts


def procedure_A(img, background, um_per_pixel, min_radius=2):

    # Read image with white soiling, black background
    microscope_img = setup.file_to_img("fake_microsope 2.tiff", background)

    # Apply masks
    otsu_mask = otsu.apply_otsu(img)
    dog_mask = apply_dog_triangle(img)
    procedure_A_mask = otsu_mask | dog_mask

    # Analyse particle count
    filled_mask = fill_outlines(procedure_A_mask)
    particle_dicts = identify_small_particles(filled_mask, um_per_pixel)
