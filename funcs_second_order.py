import numpy as np
import cv2
from pathlib import Path

import funcs as funcs1


def generate_particle_dicts(mask):
    print("generating particle dicts")
    # If either of them aren't the required files, rescan the .tiff
    # Otherwise, just output the previous files
    particle_dicts = funcs1.check_for_plk("particle_dicts.plk")
    recreated_mask = funcs1.check_for_plk("mask.plk")
    if particle_dicts is not None and recreated_mask is not None:
        print("Using cached particle data.")
        return recreated_mask, particle_dicts

    print("Cache missing; recomputing particle data.")

    # For future comparisons
    mask_og = mask.copy()

    # First 2 loops go through columns and rows (respectively) of the image, scanning for a soiled pixel.
    # While loop circles around the particle, identifying all of its pixels. Once a particle is fully identified,
    # key properties are stored in a dictionary and the particles are marked as clean.

    # Two loops -> identified the top of a particle! -> Build its outline -> Construct list of all pixels inside the particle
    # -> Identify key parameters.

    particle_list = []  # Contains each particle, which is a list of pixels

    # -2 for the borders (the outside is padding)
    for col in range(1, mask.shape[1] - 1):
        for row in range(1, mask.shape[0] - 1):
            if mask[row, col]:
                pixel_list = [(row, col)]  # List of pixels in the particle
                # Flag activated when this particle is finished.
                no_more_pixels = 0

                # Trace the outside of the particle
                while not no_more_pixels:
                    # 3x3 grid centred on current pixel
                    adjacent_map = mask[row - 1 : row + 2, col - 1 : col + 2]
                    [row, col, pixel_list, no_more_pixels] = funcs1.circle_around(
                        adjacent_map, row, col, pixel_list
                    )

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

                # Convert the numpy back to a list
                pixel_list = [tuple(p) for p in pixel_list_numpy.tolist()]

                # Clear the whole traced particle now that it's dealt with
                for r, c in pixel_list:
                    mask[r, c] = 0

                # Add the pixel_list to the particle_list
                particle_list.append(pixel_list)
                print(len(particle_list))

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
            "coords": coords,
            "centroid": None,  # (x,y)
            "effective_radius": None,
            "major_axis": None,
            "minor_axis": None,
            "orientation": None,
            "pixel_count": len(coords),
            "area": None,  # Depends on scale, could be added later
        }
        for coords in particle_list
    ]

    # Extract other information about the particles

    # Major/minor axis
    for i in range(len(particle_dicts)):
        info = funcs1.elipse_info(funcs1.centroid(particle_dicts[i]["coords"]))
        # See function elipse_info for output structure
        particle_dicts[i]["centroid"] = info[0]
        particle_dicts[i]["effective_radius"] = info[1]
        particle_dicts[i]["major_axis"] = info[2]
        particle_dicts[i]["minor_axis"] = info[3]

    # Save to Output Files/particle_dicts.pkl by default
    funcs1.save_as_plk("particle_dicts.plk", particle_dicts)
    funcs1.save_as_plk("mask.plk", mask_og)

    return recreated_mask, particle_dicts


def images_init(fake_microscope_name, border_thickness):

    print("Started initializing images")

    # Input the image as a matrix
    # 0 makes it greyscale (8 bit by default)
    img = cv2.imread(fake_microscope_name, 0)
    n = border_thickness  # Shave off the outside of the image
    img = img[n:-n, n:-n]
    funcs1.save_mat_as_image_to_output_folder(
        img, "Microscope Verification.png", "Output Files", base_dir=None
    )

    # Apply mask_image function
    mask = funcs1.mask_image(img, 255).astype(
        bool
    )  # Anything not pure white is soiled.

    # Convert the boolean mask to a temporary 8-bit image (0 or 255)
    img = ~mask  # Invert black/white
    img = img.astype(np.uint8) * 255
    funcs1.save_mat_as_image_to_output_folder(
        img, "Mask.png", "Output Files", base_dir=None
    )

    print("Finished initializing images")

    # Just return the boolean mask (not uint8)
    return mask
