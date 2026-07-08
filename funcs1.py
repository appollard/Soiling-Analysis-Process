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
    ### mask = add_border(mask) # Doesn't seem to be necessary. Makes the rows and columns offset by 1 too
    return mask


# Pad a matrix
def add_border(matrix, value):
    return np.pad(matrix, pad_width=1, mode="constant", constant_values=value)


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


# Detect the black bars in the microscope image and remove them
def crop_to_plot_area(mask: np.ndarray, min_length: int = 1000) -> np.ndarray:

    def find_bar_indices(lines: np.ndarray) -> list[int]:
        bar_indices = []
        for i, line in enumerate(lines):
            padded = np.concatenate(([0], line, [0]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            if len(starts) and np.any(ends - starts >= min_length):
                bar_indices.append(i)
        return bar_indices

    axis_rows = find_bar_indices(mask)
    axis_cols = find_bar_indices(mask.T)

    if not axis_rows or not axis_cols:
        raise ValueError(
            f"No axis bars found. "
            f"Detected {len(axis_rows)} row bars, {len(axis_cols)} col bars. "
            f"Try lowering min_length (currently {min_length})."
        )

    mid_row = mask.shape[0] // 2
    mid_col = mask.shape[1] // 2

    # Inner edge of each bar (the edge closest to the plot interior)
    row_top = max(i for i in axis_rows if i < mid_row)  # bottom edge of top bar
    row_bottom = min(i for i in axis_rows if i > mid_row)  # top edge of bottom bar
    col_left = max(i for i in axis_cols if i < mid_col)  # right edge of left bar
    col_right = min(i for i in axis_cols if i > mid_col)  # left edge of right bar

    # print(f"Axis rows: {axis_rows}")
    # print(f"Axis cols: {axis_cols}")
    print(
        f"Cropping to rows [{row_top + 1} : {row_bottom}], cols [{col_left + 1} : {col_right}]"
    )

    # +1 to exclude the inner edge pixel of the bar itself
    cropped_mask = mask[row_top + 1 : row_bottom, col_left + 1 : col_right]
    bounds = (row_top + 1, row_bottom, col_left + 1, col_right)

    return cropped_mask, bounds


# Calculate the coordinates of the centroid of the particle
# Source: "Image Moments-Based Structuring and Tracking of Objects"
def ellipse_info_method_1(
    particle, circumference, um_per_pixel, microscope_image_cropped
):
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


def ellipse_info_method_2(particle, outline, um_per_pixel, microscope_image_cropped):
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

    # Identify excess area and therefore correct diameter/area
    # Assume that excess area is a function of the greyscale of each pixel- a pixel
    # that is 0/255 is fully soiled (no excess area), 1/255 would be 1/255 of a pixel
    # less than fully soiled, etc.
    # REVISION: Replace that 255 with the average greyscale of the microscope image

    av_grey = 255  # Default
    av_grey = np.average(microscope_image_cropped[p] for p in particle)

    excess_area = 0
    for i, coord in enumerate(outline):
        greyscale_value = microscope_image_cropped[coord]
        excess_area += (greyscale_value / av_grey) * (um_per_pixel**2)  # uint8

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


# Check whether a file exists. If it does, just load it. Otherwise returns a None
def check_for_plk(file_name):
    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "Output Files" / file_name

    if file_path.is_file():
        with file_path.open("rb") as f:
            return pickle.load(f)

    return None


# Take the particles list and turn it into a picture where each particle
# is a different (to a point) colour.


def colourful_particle_map(particle_list, mask_og):

    # Define RGB colour options. Last one is tranparency
    colours = np.uint8(
        [
            (255, 0, 0, 255),
            (0, 255, 0, 255),
            (0, 0, 255, 255),
            (255, 255, 0, 255),
            (255, 0, 255, 255),
            (0, 255, 255, 255),
        ]
    )

    # Define the map
    # + (3,) gives the tensor a depth of 4: 3 for RGB and 1 for transparency
    # (default is 0 for transparent canvas, changes to 1 when colour is assigned)
    map = np.zeros((mask_og.shape + (4,)), dtype=np.uint8)

    for i in range(len(particle_list)):  # For each particle
        col = colours[i % len(colours)]  # Itterate through the colours
        for j in range(len(particle_list[i]["coords"])):
            map[particle_list[i]["coords"][j]] = col  # Sets the colour of the pixel
    return map


# Overlay this on the microscope image for comparison
def show_overlay(microscope_img, mask_og, particle_list):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    # Show the original image
    ax.imshow(microscope_img, cmap="gray", vmin=0, vmax=255)  # Assumes uint8

    # Generate the map using the particle list
    map = colourful_particle_map(particle_list, mask_og)

    # Show the map
    map_overlay = ax.imshow(map, alpha=0.4)

    # Slider to control map opacity
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Mask opacity", 0.0, 1.0, valinit=0.4)

    def update(val):
        map_overlay.set_alpha(slider.val)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


# Import the particles json as an array of triplets (d, x, y)
def import_particles_json(file_name):

    path = Path(__file__).parent / file_name
    with open(path, "r") as f:
        data = json.load(f)

    # As a numpy array of shape (N, 3) — columns are D, X, Y
    data = np.column_stack([data["D"], data["x"], data["y"]])

    return data


# Compute the cumulative volume and projected area of the circles corresponding
# to an array of diameters. diameters_array is in um, output is in m^2 and kg.
# Round to 4 sig figs
def compute_total_area_mass(diameters_array):

    diameters_array = diameters_array * 1e-6  # um -> m

    radii = diameters_array / 2
    total_area = float(f"{np.sum(np.pi * radii**2):.4g}")
    soil_density = 2000  # Silica, approx 2000kg/m^3
    total_mass = float(f"{np.sum((4/3) * np.pi * (radii**3)*soil_density):.4g}")

    return total_area, total_mass


# bmp file type to uint8 greyscale image
def bmp_to_img(file):

    img_path = Path(__file__).parent / "images" / file
    img = cv2.imread(str(img_path))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    return gray_img


# Take the JSON sata and make my own image with area-weighted aliasing
def rasterise_circles_from_json(file_name, um_per_pixel=6.5 / 60, canvas_size=7687):

    def exact_pixel_coverage(cx, cy, r, x0, x1, y0, y1):
        cols, rows = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        dist = np.sqrt((cols - cx) ** 2 + (rows - cy) ** 2)
        edge_mask = np.abs(dist - r) < 1.5

        coverage = np.zeros(cols.shape)
        # Full interior pixels
        coverage[dist <= r - 0.71] = 1.0  # 0.71 ~ sqrt(0.5), fully inside

        # Exact coverage for boundary pixels only
        circle = Point(cx, cy).buffer(r)
        edge_indices = np.argwhere(edge_mask)
        for idx in edge_indices:
            py, px = rows[idx[0], idx[1]], cols[idx[0], idx[1]]
            pixel = box(px - 0.5, py - 0.5, px + 0.5, py + 0.5)
            coverage[idx[0], idx[1]] = pixel.intersection(circle).area

        return coverage

    with open(Path(__file__).parent / file_name) as f:
        data = json.load(f)

    m_per_pixel = um_per_pixel * 1e-6
    min_r_for_exact = 2 / um_per_pixel / 2  # 2um diameter in pixels

    diameters_px = np.array(data["D"]) / m_per_pixel
    x_px = np.array(data["x"]) / m_per_pixel + canvas_size / 2
    y_px = np.array(data["y"]) / m_per_pixel + canvas_size / 2

    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255

    for i, (d, cx, cy) in enumerate(zip(diameters_px, x_px, y_px)):
        if i % 1000 == 0:
            print(f"Processing circle {i}/{len(diameters_px)}")

        r = d / 2
        cx_i, cy_i = int(round(cx)), int(round(cy))
        r_i = int(round(r))

        if r_i < 1:
            # Sub-pixel
            if 0 <= cy_i < canvas_size and 0 <= cx_i < canvas_size:
                coverage = min(1.0, (r / 0.5) ** 2)
                canvas[cy_i, cx_i] = min(canvas[cy_i, cx_i], int((1 - coverage) * 255))

        elif r >= min_r_for_exact:
            # Large particle: fast cv2
            cv2.circle(canvas, (cx_i, cy_i), max(0, r_i - 1), 0, -1)
            cv2.circle(canvas, (cx_i, cy_i), r_i, 0, 1, cv2.LINE_AA)

        else:
            # Small particle: exact shapely coverage
            x0, x1 = max(0, cx_i - r_i - 2), min(canvas_size, cx_i + r_i + 2)
            y0, y1 = max(0, cy_i - r_i - 2), min(canvas_size, cy_i + r_i + 2)
            coverage = exact_pixel_coverage(cx, cy, r, x0, x1, y0, y1)
            greyscale = ((1 - coverage) * 255).astype(np.uint8)
            patch = canvas[y0:y1, x0:x1]
            canvas[y0:y1, x0:x1] = np.minimum(patch, greyscale)

    output_path = Path(__file__).parent / "rasterised_circles.tiff"
    cv2.imwrite(str(output_path), canvas)
    return canvas
