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
import particle_analysis_funcs as paran


def procedure_A(img_name, background, um_per_pixel):

    # Read image with white soiling, black background
    microscope_img = setup.file_to_img(img_name, background)

    # Apply masks
    otsu_mask = otsu.apply_otsu(microscope_img)
    dog_mask = dog.apply_dog_triangle(microscope_img)
    procedure_A_mask = otsu_mask | dog_mask

    # Save masks
    setup.img_to_file(255 - 255 * otsu_mask, "Otsu Mask.png", "Output Files")
    setup.img_to_file(255 - 255 * dog_mask, "DoG Mask.png", "Output Files")
    setup.img_to_file(
        255 - 255 * procedure_A_mask, "Procedure A Mask.png", "Output Files"
    )

    # Analyse particle count
    filled_mask = paran.fill_outlines(procedure_A_mask)
    particle_dicts = paran.identify_particles(filled_mask, um_per_pixel)

    # Visualoise the result
    visualiser.show_overlay(microscope_img, procedure_A_mask, particle_dicts)


procedure_A("01.bmp", "black", 6.5 / 60)
