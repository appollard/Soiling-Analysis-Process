# Packages
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops


def fill_outlines(mask):
    """
    Fill the particle outlines produced by the DoG
    """

    mask_filled = binary_fill_holes(mask)

    return mask_filled


def identify_particles(mask, um_per_pixel):
    """
    Identify the particles in a mask. Returns a list of dicts- one for each particle.
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
