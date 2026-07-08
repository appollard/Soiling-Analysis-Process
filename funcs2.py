import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import math

import funcs1
import plotting
import ellipse_info_methods as ellipse


def generate_particle_dicts(
    mask, um_per_pixel, threshold, manual_regen, microscope_image_cropped, method
):
    print("generating particle dicts")
    # If either of them aren't the required files, rescan the .tiff
    # Otherwise, just output the previous files. Also check to see
    # if there's a flag for regenerating the data regardless

    # If we just use one plk file, it's overwritten each time, meaning that
    # you only have the latest when you're plotting'
    # particle_dicts = funcs1.check_for_plk(f"particle_dicts_{threshold}.plk")
    # recreated_mask = funcs1.check_for_plk(f"mask_{threshold}.plk")
    # if particle_dicts is not None and recreated_mask is not None:
    #    if not manual_regen:
    #        print("Using cached particle data.")
    #        return recreated_mask, particle_dicts
    #
    # print("Cache missing; recomputing particle data.")

    # For future comparisons
    mask_og = mask.copy()

    # First 2 loops go through columns and rows (respectively) of the image, scanning for a soiled pixel.
    # While loop circles around the particle, identifying all of its pixels. Once a particle is fully identified,
    # key properties are stored in a dictionary and the particles are marked as clean.

    # Two loops -> identified the top of a particle! -> Build its outline -> Construct list of all pixels inside the particle
    # -> Identify key parameters.

    particle_list = []  # Contains each particle, which is a list of pixels
    outline_list = []  # Contains each particle outline, which is a list of pixels
    circumference_list = []  # Contains each circumference, which is a number of pixels

    # -2 for the borders (the outside is padding)
    for col in range(1, mask.shape[1] - 1):
        for row in range(1, mask.shape[0] - 1):
            if mask[row, col]:
                print(len(particle_list))
                # List of pixels in the particle
                pixel_list = [(row, col)]
                # Flag activated when this particle is finished.
                no_more_pixels = 0

                while not no_more_pixels:
                    # 3x3 grid centred on current pixel
                    adjacent_map = mask[row - 1 : row + 2, col - 1 : col + 2]
                    [row, col, pixel_list, no_more_pixels] = funcs1.circle_around(
                        adjacent_map, row, col, pixel_list
                    )

                # Save circumference value. At this point, pixel_list is a contour, meaning that cv2.arcLength
                # will return it's circumference. Needed for Method 1.
                circumference_list.append(cv2.arcLength(np.array(pixel_list), True))

                # Save the outline (which is currently all pixel_list is)
                outline_list.append(pixel_list)

                # Fill in the inside of the particle.

                # Makes the next bits more efficient if we can use array slicing
                pixel_list_numpy = np.array(pixel_list, dtype=int)

                # Take highest/lowest column in pixel list (leftmost and rightmost boundaries of the particle)
                min_col = pixel_list_numpy[:, 1].min()
                max_col = pixel_list_numpy[:, 1].max()
                for col2 in range(min_col, max_col + 1):

                    # For a given column, scan between its highest and lowest particles in pixel_list
                    column_pixels = pixel_list_numpy[pixel_list_numpy[:, 1] == col2]
                    if (
                        column_pixels.size == 0
                    ):  # Shouldn't be necessary but just in case
                        continue
                    min_row = column_pixels[:, 0].min()
                    max_row = column_pixels[:, 0].max()
                    for row2 in range(min_row, max_row + 1):
                        if mask[row2, col2]:
                            pixel_list_numpy = np.vstack(
                                (pixel_list_numpy, np.array([[row2, col2]], dtype=int))
                            )

                # Remove any duplicates, should only be from path tracing
                pixel_list = np.unique(pixel_list_numpy, axis=0)

                # Convert the numpy back to a list
                pixel_list = [tuple(p) for p in pixel_list_numpy.tolist()]

                # Clear the whole traced particle now that it's dealt with
                for r, c in pixel_list:
                    mask[r, c] = 0

                # Add the pixel_list to the particle_list
                particle_list.append(pixel_list)
                # print(len(particle_list))

    # Turn the particle list back into a mask
    recreated_mask = funcs1.particle_list_to_bool_array(particle_list, mask_og)
    funcs1.save_mat_as_image_to_output_folder(
        recreated_mask, "Mask Verification.png", "Output Files", base_dir=None
    )

    # Check the recreation
    funcs1.check_mask(~recreated_mask, mask_og)

    # Turn the particles list into a dict
    particle_dicts = [
        {
            "coords": coords,  # pixels units
            "outline_coords": outline,  # pixel units
            "circumference": circumference,  # pixel units
            "centroid": None,  # (x,y) assigned um units later
            "effective_diameter": None,  # assigned um units later
            "major_axis": None,  # assigned um units later
            "minor_axis": None,  # assigned um units later
            "orientation": None,
            "pixel_count": len(coords),
            "area": len(coords)
            * (um_per_pixel**2),  # Depends on scale, could be added later
            "corrected_diameter": None,
            "corrected_area": None,
        }
        for coords, circumference, outline in zip(
            particle_list, circumference_list, outline_list
        )
    ]

    # Extract other information about the particles. Two possible methods
    for i in range(len(particle_dicts)):
        if method == "Method 1":

            info = ellipse.method_1(
                particle_dicts[i]["coords"],
                particle_dicts[i]["circumference"],
                particle_dicts[i]["outline_coords"],
                um_per_pixel,
                microscope_image_cropped,
            )
        elif method == "Method 2_1":
            info = ellipse.method_2_1(
                particle_dicts[i]["coords"],
                particle_dicts[i]["circumference"],
                particle_dicts[i]["outline_coords"],
                um_per_pixel,
                microscope_image_cropped,
            )
        elif method == "Method 2_2":
            info = ellipse.method_2_2(
                particle_dicts[i]["coords"],
                particle_dicts[i]["circumference"],
                particle_dicts[i]["outline_coords"],
                um_per_pixel,
                microscope_image_cropped,
            )
        elif method == "Method 2_3":
            info = ellipse.method_2_3(
                particle_dicts[i]["coords"],
                particle_dicts[i]["circumference"],
                particle_dicts[i]["outline_coords"],
                um_per_pixel,
                microscope_image_cropped,
            )

        # See function elipse_info for output structure
        particle_dicts[i]["centroid"] = info[0]  # in um
        particle_dicts[i]["effective_diameter"] = info[1]  # in um
        particle_dicts[i]["major_axis"] = info[2]  # in um
        particle_dicts[i]["minor_axis"] = info[3]  # in um
        particle_dicts[i]["corrected_diameter"] = info[5]  # in um
        particle_dicts[i]["corrected_area"] = info[6]  # in um

    # Save to Output Files/particle_dicts.pkl by default
    funcs1.save_as_plk("particle_dicts.plk", particle_dicts)
    funcs1.save_as_plk("mask.plk", mask_og)

    return recreated_mask, particle_dicts


def image_to_mask_and_save(img, thresh):

    print("Started initializing images")

    # Apply mask_image function
    mask = funcs1.mask_image(img, thresh)  # Anything not pure white is soiled.

    # Return the cropped mask AND cropping boundaries so we can crop the microscope image
    # identically
    mask, (r0, r1, c0, c1) = funcs1.crop_to_plot_area(mask, min_length=1000)
    img = img[r0:r1, c0:c1]
    mask = mask.astype(bool)
    # mask = funcs1.add_border(mask)

    # Crop and save the image
    funcs1.save_mat_as_image_to_output_folder(
        img, "Microscope Verification.png", "Output Files", base_dir=None
    )

    # Convert the boolean mask to a temporary 8-bit image (0 or 255)
    img2 = ~mask  # Invert black/white
    img2 = img2.astype(np.uint8) * 255
    funcs1.save_mat_as_image_to_output_folder(
        img2, "Mask.png", "Output Files", base_dir=None
    )

    print("Finished initializing images")

    # Just return the boolean mask (not uint8)
    return img, mask


def image_to_mask_and_save_no_crop(img, thresh):

    print("Started initializing images")

    # Apply mask_image function
    mask = funcs1.mask_image(img, thresh)  # Anything not pure white is soiled.

    # Save the image
    funcs1.save_mat_as_image_to_output_folder(
        img, "Microscope Verification.png", "Output Files", base_dir=None
    )

    mask = funcs1.add_border(mask, False)  # Add a clean border around the mask
    img = funcs1.add_border(img, 255)  # Add a clean border around the image

    # Convert the boolean mask to a temporary 8-bit image (0 or 255)
    img2 = ~mask  # Invert black/white
    img2 = img2.astype(np.uint8) * 255
    funcs1.save_mat_as_image_to_output_folder(
        img2, "Mask.png", "Output Files", base_dir=None
    )

    print("Finished initializing images")

    # Just return the boolean mask (not uint8)
    return img, mask
