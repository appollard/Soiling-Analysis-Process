import os
import cv2
import numpy as np
import math
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import json
from shapely.geometry import box, Point


def ellipse_setup(particle, um_per_pixel):

    x = [p[1] for p in particle]  # List of x coords
    y = [p[0] for p in particle]  # List of y coords

    m00 = len(particle)  # Number of pixels

    m10 = sum(x)  # Sum of x coords
    m01 = sum(y)  # Sum of y coords

    m11 = sum(xi * yi for xi, yi in zip(x, y))  # 2nd order xy moment
    m20 = sum(xi * xi for xi in x)  # 2nd order x moment
    m02 = sum(yi * yi for yi in y)  # 2nd order y moment

    # Uses the centroid information to identify the major and minor axes of the equivalent elipse.
    # Units are um
    # Source: "Image Moments-Based Structuring and Tracking of Objects"

    # Establish key parameters
    # 1/12's are for Sheppard's correction- since each pixel is a square, not a singularity,
    # we should account for this. Otherwise we're a bit off, and single-pixel particles don't
    # work at all. For single-pixel particles, you get minor and major axes of 1, and thus an
    # effective diameter of 1.155. This is greater than 1 since the moment of inertia of a
    # square exceeds that of an ellipse

    xc = m10 / m00  # x axis centroid
    yc = m01 / m00  # y axis centroid
    # Continuous-space normalized second-order central moment along the x-axis
    a = m20 / m00 - xc**2 + 1 / 12
    # 2 * normalized central mixed moment
    b = 2 * (m11 / m00 - xc * yc)
    # Continuous-space normalized second-order central moment along the y-axis
    c = m02 / m00 - yc**2 + 1 / 12

    # Calculate them. Major and minor are radial.
    # sqrt(minor*major) = radius*sqrt(3). To account for this, divide by sqrt(3).
    # This is the same as replacing the 6 in the formula from the paper with an 8
    # and neglecting the sqrt(3)

    minor = math.sqrt(8 * (a + c - math.sqrt(b**2 + (a - c) ** 2))) * um_per_pixel
    major = math.sqrt(8 * (a + c + math.sqrt(b**2 + (a - c) ** 2))) * um_per_pixel
    effective_diameter = math.sqrt(minor * major)

    xc = xc * um_per_pixel
    yc = yc * um_per_pixel

    area = m00 * (um_per_pixel**2)  # num pixels (m00) * um^2 per pixel = area

    return [area, minor, major, effective_diameter, (xc, yc)]


# Add a number of pixels per unit circumference
def method_1(particle, circumference, outline, um_per_pixel, microscope_image_cropped):

    area, minor, major, effective_diameter, (xc, yc) = ellipse_setup(
        particle, um_per_pixel
    )

    # Identify excess area and therefore correct diameter/area
    # Assume that excess area 1 pixel per pixel side-length of circumference
    excess_per_pixel = 1  # Assume due to severe masking

    # Excess per pixel is the minimum of 1 (can't be higher for a particle with a
    # high circumference to area ratio) and the defined value.
    excess_area = min(excess_per_pixel, 1) * circumference * (um_per_pixel**2)
    corrected_area = area - excess_area
    corrected_diameter = 2 * math.sqrt(corrected_area / math.pi)

    return [
        (xc, yc),
        effective_diameter,
        major,
        minor,
        area,
        corrected_diameter,
        corrected_area,
    ]


# Compare every pixel against black
def method_2_1(
    particle, circumference, outline, um_per_pixel, microscope_image_cropped
):

    area, minor, major, effective_diameter, (xc, yc) = ellipse_setup(
        particle, um_per_pixel
    )

    # Identify excess area and therefore correct diameter/area
    # Assume that excess area is a function of the greyscale of each pixel- a pixel
    # that is 0/255 is fully soiled (no excess area), 1/255 would be 1/255 of a pixel
    # less than fully soiled, etc.

    av_grey = 0  # Default

    excess_area = 0
    for i, coord in enumerate(particle):
        greyscale_value = microscope_image_cropped[coord]
        excess_area += (greyscale_value / (255 - av_grey)) * (um_per_pixel**2)  # uint8

    # excess_per_pixel = 1  # Assume due to severe masking (currently insufficient)

    # Excess per pixel is the minimum of 1 (can't be higher for a particle with a
    # high circumference to area ratio) and the defined value.
    # excess_area = min(excess_per_pixel, 1) * circumference * (um_per_pixel**2)
    corrected_area = area - excess_area
    corrected_diameter = 2 * math.sqrt(corrected_area / math.pi)

    return [
        (xc, yc),
        effective_diameter,
        major,
        minor,
        area,
        corrected_diameter,
        corrected_area,
    ]


# Compare each outline pixel against the average of the particle
def method_2_2(
    particle, circumference, outline, um_per_pixel, microscope_image_cropped
):

    area, minor, major, effective_diameter, (xc, yc) = ellipse_setup(
        particle, um_per_pixel
    )

    # Identify excess area and therefore correct diameter/area
    # Assume that excess area is a function of the greyscale of each pixel- a pixel
    # that is 0/255 is fully soiled (no excess area), 1/255 would be 1/255 of a pixel
    # less than fully soiled, etc.
    # REVISION: Replace that 255 with the average greyscale of the microscope image

    av_grey = 255  # Default
    av_grey = np.mean([microscope_image_cropped[p] for p in particle])
    excess_area = 0
    for i, coord in enumerate(outline):
        greyscale_value = microscope_image_cropped[coord]
        # Could be negative if the outline happens to be darker than the average. The max(0,-) accounts for this
        excess_area += max(
            0,
            min(1, 2 * ((greyscale_value - av_grey) / (255 - av_grey)))
            * (um_per_pixel**2),
        )  # uint8

    # If
    corrected_area = area - excess_area
    corrected_diameter = 2 * math.sqrt(corrected_area / math.pi)

    return [
        (xc, yc),
        effective_diameter,
        major,
        minor,
        area,
        corrected_diameter,
        corrected_area,
    ]


# Compare each pixel against the average of the particle
def method_2_3(
    particle, circumference, outline, um_per_pixel, microscope_image_cropped
):

    area, minor, major, effective_diameter, (xc, yc) = ellipse_setup(
        particle, um_per_pixel
    )

    # Identify excess area and therefore correct diameter/area
    # Assume that excess area is a function of the greyscale of each pixel- a pixel
    # that is 0/av is fully soiled (no excess area), 1/av would be 1/av of a pixel
    # less than fully soiled, etc.

    av_grey = 0  # Default
    av_grey = np.mean([microscope_image_cropped[p] for p in particle])

    excess_area = 0
    for i, coord in enumerate(particle):
        # Maps greyscale value based on the average of the particle. It is now a proportion.
        greyscale_value = max(0, microscope_image_cropped[coord] - av_grey)
        excess_area += (greyscale_value / (255 - av_grey)) * (um_per_pixel**2)  # uint8

    corrected_area = area - excess_area
    corrected_diameter = 2 * math.sqrt(corrected_area / math.pi)

    return [
        (xc, yc),
        effective_diameter,
        major,
        minor,
        area,
        corrected_diameter,
        corrected_area,
    ]
