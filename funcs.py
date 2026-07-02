import os
import cv2
import numpy as np
import math
import pickle
from pathlib import Path


def save_mat_as_image_to_output_folder(img, filename, folder_name, base_dir=None):
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
    return output_path


# Define a function to turn the image into a mask
def mask_image(image, thresh):
    mask = (
        image[:, :] < thresh
    )  # If each bit is lower (less white) than this threshold, it will be flagged as being soiled (1).
    mask = add_border(mask)
    return mask


# Pad a matrix
def add_border(matrix):
    return np.pad(matrix, pad_width=1, mode="constant", constant_values=False)


# Define a function which, when given a 3x3 grid about a centre, identifies the next pixel to move to
def circle_around(adjacent_map, row, col, pixel_list):

    # Adjacent_map is a 3x3 window around the current pixel
    # Build a flattened neighbour array in the order:
    # centre, left, down-left, down, down-right, right, up-right, up, up-left
    adjacent_map_array = np.array(
        [
            adjacent_map[1, 0],  # left
            adjacent_map[2, 0],  # down-left
            adjacent_map[2, 1],  # down
            adjacent_map[2, 2],  # down-right
            adjacent_map[1, 2],  # right
            adjacent_map[0, 2],  # up-right
            adjacent_map[0, 1],  # up
            adjacent_map[0, 0],  # up-left
        ],
        dtype=bool,
    )

    # Now we determine the direction of the previous move. If it's the first pixel,
    # we skip this.
    back_index = None
    if len(pixel_list) > 1:  # Only ==2 if there's on soiled other than the centre
        prev_row, prev_col = pixel_list[-2]  # -1 would be the current pixel
        offset = (
            prev_row - row,
            prev_col - col,
        )  # Gets the relationship between current and last pixel
        back_index = {  # Finds the index of the last pixel in terms of the
            (0, -1): 1,  # adjacent_map_array structure, so that we can make sure
            (1, -1): 2,  # it isn't the same as the next pixel we choose
            (1, 0): 3,
            (1, 1): 4,
            (0, 1): 5,
            (-1, 1): 6,
            (-1, 0): 7,
            (-1, -1): 8,
        }.get(offset)

    # Rotate the list so that the previous pixel direction is at index 0.
    # Then start the search from the following element.
    if back_index is not None:
        order = []
        for i in range(1, 9):
            idx = ((back_index + i - 1) % 8) + 1
            if idx != back_index:
                order.append(idx)
        order.append(back_index)  # previous pixel is the last option
    else:
        order = [1, 2, 3, 4, 5, 6, 7, 8]  # Default- start with centre left

    # Pick the first neighbour in that order that is soiled.
    last_index_soiled = None
    for idx in order:
        if adjacent_map_array[idx - 1]:
            last_index_soiled = idx
            break

    # If no neighbour exists, stop tracing.
    if last_index_soiled is None:
        no_more_pixels = 1
        return [row, col, pixel_list, no_more_pixels]

    # Move to the selected neighbour.
    match last_index_soiled:
        case 0:
            row = row
            col = col
        case 1:
            row = row
            col = col - 1
        case 2:
            row = row + 1
            col = col - 1
        case 3:
            row = row + 1
            col = col
        case 4:
            row = row + 1
            col = col + 1
        case 5:
            row = row
            col = col + 1
        case 6:
            row = row - 1
            col = col + 1
        case 7:
            row = row - 1
            col = col
        case 8:
            row = row - 1
            col = col - 1
    pixel_list.append((row, col))

    # If adding the last element means we duplicate the first element, we're done.
    # We can duplicate other elements however (like if a particle has a long tail
    # that the searcher has to double back on.)
    if pixel_list[-1] == pixel_list[0]:
        pixel_list.pop()  # Remove duplicate
        no_more_pixels = 1
    else:
        no_more_pixels = 0

    return [row, col, pixel_list, no_more_pixels]


# Verify that the pic->particle conversion worked by turning the particles back into an image
def particle_list_to_bool_array(particle_list, mask_og):

    pic = np.ones(mask_og.shape, dtype=bool)

    for i in range(len(particle_list)):
        for j in range(len(particle_list[i])):
            pic[particle_list[i][j]] = (
                False  # Paints the coordinates in the particles list as soiled.
            )
    return pic


def save_as_plk(
    file_name, file, filepath=None, folder_name="Output Files", base_dir=None
):
    if base_dir is None:
        base_dir = Path(__file__).parent

    if filepath is None:
        path = base_dir / folder_name / file_name
    else:
        path = Path(filepath)
        # If a relative filename without parent specified, place it in the output folder
        if path.parent == Path("") or path.parent == Path("."):
            path = base_dir / folder_name / path
        else:
            # Ensure provided parent directories exist
            path.parent.mkdir(parents=True, exist_ok=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(path)


# Check whether the created mask is right
def check_mask(v1, v2):
    if np.array_equal(np.asarray(v1), np.asarray(v2)):
        print("The recreated mask is correct.")
    else:
        print("The recreated mask is incorrect.")
        error_mask = [v1 == v2]
        save_mat_as_image_to_output_folder(
            error_mask,
            "Error Mask",
            "Output Files",
        )


# Calculate the coordinates of the centroid of the particle
# Source: "Image Moments-Based Structuring and Tracking of Objects"
def centroid(particle):
    x = [p[1] for p in particle]  # List of x coords
    y = [p[0] for p in particle]  # List of y coords

    m00 = len(particle)  # Number of pixels

    m10 = sum(x)  # Sum of x coords
    m01 = sum(y)  # Sum of y coords

    m11 = sum(xi * yi for xi, yi in zip(x, y))  # 2nd order xy moment
    m20 = sum(xi * xi for xi in x)  # 2nd order x moment
    m02 = sum(yi * yi for yi in y)  # 2nd order y moment

    return [m00, m10, m01, m11, m20, m02]


# Uses the centroid information to identify the major and minor axes of the equivalent elipse
# Source: "Image Moments-Based Structuring and Tracking of Objects"
def elipse_info(input):
    [m00, m10, m01, m11, m20, m02] = input

    # Establish key parameters
    xc = m10 / m00  # x axis centroid
    yc = m01 / m00  # y axis centroid
    a = m20 / m00 - xc**2  # simplifying constant
    b = 2 * (m11 / m00 - xc * yc)  # simplifying constant
    c = m02 / m00 - yc**2  # simplifying constant

    # Calculate them
    minor = math.sqrt(6 * (a + c - math.sqrt(b**2 + (a - c) ** 2)))
    major = math.sqrt(6 * (a + c + math.sqrt(b**2 + (a - c) ** 2)))
    effective_radius = math.sqrt(minor * major)

    return [(xc, yc), effective_radius, major, minor]


# Check whether a file exists. If it does, just load it. Otherwise returns a None
def check_for_plk(file_name):
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "Output Files" / file_name

    if file_path.is_file():
        with file_path.open("rb") as f:
            return pickle.load(f)

    return None
