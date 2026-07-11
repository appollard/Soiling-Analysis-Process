import numpy as np
import os
import cv2
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter, binary_erosion, distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def apply_otsu(img):
    """
    Generate a mask for the image with a threshold determined by the otsu function.
    This minimizes the variation between the greyscale values of 'clean' and 'soiled'
    particles. This means that it ignores smaller particles.
    """

    thresh = threshold_otsu(img)
    mask = img < thresh

    return mask


def identify_large_particles(mask, img, min_radius, um_per_pixel):
    """
    Identify the large particles in the mask produced by apply_otsu. Returns a list of
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


def remove_particles(img, particle_dicts, noise_mean, noise_sd):
    """
    Take the original microscope image and replace all pixels which contain identified particles
    with the average grey of the image plus noise. this is crude and assumes low particle density.
    """

    new_img = img.copy()

    # Generate a full noise field and smooth it
    noise = np.random.normal(noise_mean, noise_sd, img.shape)

    for p in particle_dicts:
        for row, col in p["coords"]:
            new_img[row, col] = noise[row, col]
    return new_img


def save_otsu_product(
    img, filename, folder_name="Output Files/otsu stage images", base_dir=None
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


def remove_halos(img, noise_mean, particle_dicts, sigma=3, strength=60):
    """
    Remove halos from around each particle. Cut out a box around each particle that's
    5 sd'd or 2 pixels, whichever's larger. make a mask which is all the soiled pixels
    in this box. Erode its outside by one pixel. Take the difference of the two, leaving
    just the boundary of the particle. Apply a gausian blur to the ring of 1's. Zero out
    the interior, and then add the result to the image. Then clip it all to the average
    background grey.
    """

    img_out = img.astype(float).copy()

    for p in particle_dicts:

        coords = np.array(p["coords"])
        pad = max(int(sigma * 5), 2)  # Max out at 5 s.d's from the particle boundary
        r0 = max(0, coords[:, 0].min() - pad)
        r1 = min(img.shape[0], coords[:, 0].max() + pad)
        c0 = max(0, coords[:, 1].min() - pad)
        c1 = min(img.shape[1], coords[:, 1].max() + pad)

        if r1 <= r0 or c1 <= c0:
            continue  # If we're right on the edge this saves us

        # Local particle mask
        local_mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        for row, col in coords:
            local_mask[row - r0, col - c0] = True

        # Boundary = particle pixels with at least one non-particle neighbour
        interior = binary_erosion(local_mask)
        boundary = local_mask & ~interior

        # Convolve boundary mask with Gaussian — equivalent to summing
        # a Gaussian hill from every boundary pixel, but fast
        boundary_float = boundary.astype(float)
        correction = gaussian_filter(boundary_float, sigma=sigma)
        correction[local_mask] = 0  # only apply outside particle
        if correction.max() > 0:
            correction /= correction.max()

        img_out[r0:r1, c0:c1] += correction * strength

    return np.clip(img_out, 0, noise_mean).astype(np.uint8)


def remove_halos_multistage(img, noise_mean, particle_dicts):
    """
    Perform multiple rounds of removing halos, each one for a different effect.
    """
    # Strength arrays from trialing
    s0 = [400, 50, 8, 8]
    s1 = [400, 60, 12, 12]  # Tunes for 01.bmp bottom particle
    # Stage 1: diffraction fringe — tight, strong
    img = remove_halos(img, noise_mean, particle_dicts, sigma=1, strength=400)

    # Stage 2: scattering — medium
    img = remove_halos(img, noise_mean, particle_dicts, sigma=3, strength=60)

    # Stage 3: PSF blur — broad, gentle
    img = remove_halos(img, noise_mean, particle_dicts, sigma=10, strength=12)

    # Stage 4: PSF blur — broad, gentle duplicate
    img = remove_halos(img, noise_mean, particle_dicts, sigma=15, strength=12)

    return img


def remove_halos_adaptive(img, noise_mean, noise_sd, particle_dicts, sigma=3):
    """
    Adaptive halo removal — measures actual halo depth per particle
    and scales correction to bring halo pixels back to background level.
    """
    img_out = img.astype(float).copy()

    for p in particle_dicts:
        coords = np.array(p["coords"])
        pad = max(int(sigma * 5), 2)
        r0 = max(0, coords[:, 0].min() - pad)
        r1 = min(img.shape[0], coords[:, 0].max() + pad)
        c0 = max(0, coords[:, 1].min() - pad)
        c1 = min(img.shape[1], coords[:, 1].max() + pad)

        if r1 <= r0 or c1 <= c0:
            continue

        # Local particle mask
        local_mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        for row, col in coords:
            local_mask[row - r0, col - c0] = True

        # Sample the halo region — pixels just outside the particle
        from scipy.ndimage import binary_dilation

        halo_ring = binary_dilation(local_mask, iterations=2) & ~local_mask

        if halo_ring.any():
            halo_pixels = img_out[r0:r1, c0:c1][halo_ring]
            halo_mean = halo_pixels.min()

            # How far below background is the halo?
            halo_depth = noise_mean - halo_mean

            # Skip if halo is within 1 sd of background — nothing to correct
            if halo_depth < noise_sd:
                continue

            # Scale strength to exactly close the gap to background
            strength = halo_depth
        else:
            continue

        # Boundary kernel
        interior = binary_erosion(local_mask)
        boundary = local_mask & ~interior
        boundary_float = boundary.astype(float)
        correction = gaussian_filter(boundary_float, sigma=sigma)
        correction[local_mask] = 0
        if correction.max() > 0:
            correction /= correction.max()

        img_out[r0:r1, c0:c1] += correction * strength

    return np.clip(img_out, 0, noise_mean).astype(np.uint8)


def remove_halos_multistage_adaptive(img, noise_mean, noise_sd, particle_dicts):
    img = remove_halos_adaptive(
        img, noise_mean, noise_sd, particle_dicts, sigma=1
    )  # Is this as weak as it should be?
    img = remove_halos_adaptive(img, noise_mean, noise_sd, particle_dicts, sigma=3)
    img = remove_halos_adaptive(img, noise_mean, noise_sd, particle_dicts, sigma=10)
    img = remove_halos_adaptive(img, noise_mean, noise_sd, particle_dicts, sigma=15)
    return img


def full_otsu_stage(img, min_radius, um_per_pixel, noise_mean, noise_sd):
    """
    Take the image, apply the otsu threshold, identify the particles, remove them, remove
    the halos and then save the old image and each stage of the new image.
    """

    mask = apply_otsu(img)
    particle_dicts = identify_large_particles(mask, img, min_radius, um_per_pixel)

    new_img = remove_particles(img, particle_dicts, noise_mean, noise_sd)
    no_halo_img = remove_halos_multistage_adaptive(
        new_img, noise_mean, noise_sd, particle_dicts
    )
    save_otsu_product(img, "Original image.png")
    save_otsu_product(new_img, "Large particles removed.png")
    save_otsu_product(255 - (255 * mask), "otsu Mask.png")
    save_otsu_product(no_halo_img, "Halo removed.png")

    return particle_dicts, no_halo_img, mask
