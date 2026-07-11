import matplotlib

print(matplotlib.get_backend())

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json


def add_gaussian_noise(img, mean=0, std=5):
    """
    Add Gaussian noise to an (idealised) image.
    std controls the intensity — 5 is subtle, 20 is visible.
    """
    noise = np.random.normal(mean, std, img.shape)
    noisy = img.astype(float) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def import_particles_json(file_name):

    path = Path(__file__).parent / file_name
    with open(path, "r") as f:
        data = json.load(f)

    # As a numpy array of shape (N, 3) — columns are D, X, Y
    data = np.column_stack([data["D"], data["x"], data["y"]])

    return data


def plot_diameters_measured_true_comparison(measured, true):

    # Make the text bigger
    plt.rcParams.update(
        {
            "font.size": 28,
            "axes.titlesize": 32,
            "axes.labelsize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
        }
    )

    # Make sure the inputs are sorted
    measured = np.sort(measured)
    true = np.sort(true)

    # Plot
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(24, 13.5))

    title = "Placeholder.png"
    fig.suptitle(title, fontsize=32, y=1.01)

    # Plotting niceties
    combined_min = min(true.min(), measured.min())
    combined_max = max(true.max(), measured.max())
    bins = np.linspace(combined_min, combined_max, 201)  # 200 bins, 201 edges

    # Original plot
    ax[0].hist(measured, bins=bins, log=True, alpha=0.6, label="Measured")
    ax[0].hist(true, bins=bins, log=True, alpha=0.6, label="True")
    ax[0].set_xlim((combined_min, combined_max))
    ax[0].set_xlabel("Particle diameter (um)")
    ax[0].set_ylabel("Number of particles")
    ax[0].set_title("Unmodified Frequency Plot of Particle Diameter")
    ax[0].legend()

    # Note saying number of measured/true particles
    ax[0].text(
        0.99,
        0.99,
        f"Measured: {len(measured)}\nTrue: {len(true)}",
        transform=ax[0].transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Compute area/mass of measured/true soiling, assuming perfectly accurate diameters
    measured_area = np.sum(np.pi * (measured / 2) ** 2)
    measured_volume = np.sum(4 / 3 * np.pi * (measured / 2) ** 3)
    true_area = np.sum(np.pi * (true / 2) ** 2)
    true_volume = np.sum(4 / 3 * np.pi * (true / 2) ** 3)
    area_proportion = f"{measured_area / true_area * 100:.4g}"  # %
    mass_proportion = f"{measured_volume / true_volume * 100:.4g}"  # %

    # Note saying the measured/true area/mass and proportion
    ax[0].text(
        0.5,
        0.9,
        f"Measured area proportion: {area_proportion}\nMeasured mass proportion: {mass_proportion}",
        transform=ax[0].transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Derive an empirical linear scaling factor between the measured and true diameters.
    max_measured = measured[-1]
    max_true = true[-1]
    empirical_scaling_factor = max_true / max_measured
    scaled_measured = measured * empirical_scaling_factor
    print(
        f"Max measured: {max_measured:.4f}um, Max true: {max_true:.4f}um, Empirical scaling factor: {empirical_scaling_factor:.4f}"
    )

    # Scaled plot
    ax[1].hist(scaled_measured, bins=bins, log=True, alpha=0.6, label="Measured")
    ax[1].hist(true, bins=bins, log=True, alpha=0.6, label="True")
    ax[1].set_xlim((combined_min, combined_max))
    ax[1].set_xlabel("Particle diameter (um)")
    ax[1].set_ylabel("Number of particles")
    ax[1].set_title("Frequency Plot of Particle Diameter Multiplied By Matching Factor")
    ax[1].legend()

    # Compute area/mass of scaled measured soiling, assuming perfectly accurate diameters
    scaled_measured_area = np.sum(np.pi * (scaled_measured / 2) ** 2)
    scaled_measured_volume = np.sum(4 / 3 * np.pi * (scaled_measured / 2) ** 3)
    scaled_area_proportion = f"{scaled_measured_area / true_area * 100:.4g}"  # %
    scaled_mass_proportion = f"{scaled_measured_volume / true_volume * 100:.4g}"  # %

    # Note saying the measured/true area/mass and proportion
    ax[1].text(
        0.5,
        0.9,
        f"Measured area proportion: {scaled_area_proportion}\nMeasured mass proportion: {scaled_mass_proportion}",
        transform=ax[1].transAxes,
        ha="center",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Determine CDF at different diameter points to see if there's a skew
    d_points = np.linspace(2, combined_max, 500)
    measured_cdf = np.searchsorted(measured[measured > 2], d_points) / len(
        measured[measured > 2]
    )
    true_cdf = np.searchsorted(true[true > 2], d_points) / len(true[true > 2])

    # Plot ratio of CDFs
    ax[2].plot(
        d_points, np.log10(measured_cdf / (true_cdf + 1e-10))
    )  # avoid div by zero
    ax[2].set_xlabel("Particle diameter (um)")
    ax[2].set_ylabel(r"$\log_{10}\left(\frac{\mathrm{measured}}{\mathrm{true}}\right)$")
    ax[2].set_title(
        "Ratio Between Measured and True Cumulative Distributions For Particles Over 2um"
    )
    ax[2].axhline(0, color="r", linestyle="--")

    plt.tight_layout()
    # plt.get_current_fig_manager().window.state("zoomed")
    output_path = Path(__file__).parent / "Output Files" / "Testing Plots" / title
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(output_path)

    plt.show()

    return


def compare_measured_to_true_dist(json_name, particle_dicts):

    true_diameter_list = np.sort(import_particles_json(json_name)[:, 0])

    measured_diameter_list = np.sort(
        np.array([p["effective_diameter"] for p in particle_dicts])
    )

    # Get rid of the big particle that might not be supposed to be there?
    true_diameter_list = true_diameter_list[:-1]
    measured_diameter_list = measured_diameter_list[:-1]

    plot_diameters_measured_true_comparison(measured_diameter_list, true_diameter_list)
