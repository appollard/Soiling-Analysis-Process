import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

import old.funcs1 as funcs1
import old.funcs2 as funcs2


def plot_diameters_measured_true_comparison(measured, true, threshold, method):

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

    title = "Mask Threshold " + str(threshold) + " " + method
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
    measured_area, measured_mass = funcs1.compute_total_area_mass(measured)
    true_area, true_mass = funcs1.compute_total_area_mass(true)
    area_proportion = f"{measured_area / true_area * 100:.4g}"  # %
    mass_proportion = f"{measured_mass / true_mass * 100:.4g}"  # %

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
    scaled_measured_area, scaled_measured_mass = funcs1.compute_total_area_mass(
        scaled_measured
    )
    scaled_area_proportion = f"{scaled_measured_area / true_area * 100:.4g}"  # %
    scaled_mass_proportion = f"{scaled_measured_mass / true_mass * 100:.4g}"  # %

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
    plt.get_current_fig_manager().window.state("zoomed")
    output_path = Path(__file__).parent / "Output Files" / "Mask Thresholds" / title
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show(block=False)

    return (
        float(area_proportion),
        float(mass_proportion),
        float(scaled_area_proportion),
        float(scaled_mass_proportion),
    )  # str by default


def plot_multiple_threshold_scenarios_test(
    threshold_list, microscope_img, interactive_thresh, true_diameter_json, method
):

    area_mass_ratio_tuples = dict.fromkeys(threshold_list)
    for threshold in threshold_list:

        # Turn the microscope image and threshold into a mask and save it
        img, mask = funcs2.image_to_mask_and_save(microscope_img, threshold)

        # Generate a list of dicts, where each dict contains info about the particles
        # Also generate a recreation of the mask to verify that it all worked. Manual
        # regen to ensure that new particles are generated each time.
        recreated_mask, particle_dicts = funcs2.generate_particle_dicts(
            mask, 6.5 / 60, threshold, True, img, method
        )

        true_diameter_list = np.sort(
            funcs1.import_particles_json(true_diameter_json)[:, 0]
        )
        print(
            f"True diameter list length: {len(true_diameter_list)}, min: {true_diameter_list.min()}, max: {true_diameter_list.max()}"
        )
        diameter_list = np.sort(
            np.array([p["corrected_diameter"] for p in particle_dicts])
        )
        area_mass_ratio_tuples[threshold] = plot_diameters_measured_true_comparison(
            diameter_list, true_diameter_list, threshold, method
        )

        # Make interactive environment that lets you see all particles overlayed the original image.
        # Only for the chosen threshold (so you know what you're looking at)
        if threshold == interactive_thresh:
            funcs1.show_overlay(
                img,
                mask,
                particle_dicts,
            )

        # plt.close("all")

    return area_mass_ratio_tuples


def plot_multiple_threshold_scenarios_real(
    threshold_list, microscope_img, interactive_thresh, method
):

    for threshold in threshold_list:

        # Turn the microscope image and threshold into a mask and save it
        img, mask = funcs2.image_to_mask_and_save_no_crop(microscope_img, threshold)

        # Generate a list of dicts, where each dict contains info about the particles
        # Also generate a recreation of the mask to verify that it all worked. Manual
        # regen to ensure that new particles are generated each time.
        recreated_mask, particle_dicts = funcs2.generate_particle_dicts(
            mask, 6.5 / 60, threshold, True, img, method
        )
        print(np.shape(mask))
        diameter_list = np.sort(
            np.array([p["corrected_diameter"] for p in particle_dicts])
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 13.5))

        bins = 201
        ax.hist(diameter_list, bins=bins, log=True, alpha=0.6, label="Measured")
        ax.set_xlim((min(diameter_list), max(diameter_list)))
        ax.set_xlabel("Particle diameter (um)")
        ax.set_ylabel("Number of particles")
        ax.set_title("Unmodified Frequency Plot of Particle Diameter")
        ax.legend()
        plt.show()

        # Make interactive environment that lets you see all particles overlayed the original image.
        # Only for the chosen threshold (so you know what you're looking at)
        if threshold == interactive_thresh:
            funcs1.show_overlay(
                img,
                mask,
                particle_dicts,
            )

        plt.close("all")

    return


# Make two grouped bar charts. One for area, one for mass. Group by threshold, each group has both methods
# with multiplicative factor and without.
# Input is two dictionaries with thresholds as the keys, combined into a list so that more method can be
# used in the future
def grouped_bar_thresholds_methods(main_title, method_dicts):
    groups = list(method_dicts[0].keys())
    n_groups = len(groups)
    n_methods = len(method_dicts)
    n_bars = n_methods * 2  # scaled and unscaled per method

    x = np.arange(n_groups)
    width = 0.8 / n_bars

    fig, (ax_area, ax_mass) = plt.subplots(1, 2, figsize=(16, 5))

    for i, method_dict in enumerate(method_dicts):
        # Unscaled: indices 0, 1; Scaled: indices 2, 3
        unscaled_areas = [float(method_dict[g][0]) for g in groups]
        unscaled_masses = [float(method_dict[g][1]) for g in groups]
        scaled_areas = [float(method_dict[g][2]) for g in groups]
        scaled_masses = [float(method_dict[g][3]) for g in groups]

        offset_unscaled = (i * 2 - n_bars / 2 + 0.5) * width
        offset_scaled = (i * 2 + 1 - n_bars / 2 + 0.5) * width

        ax_area.bar(
            x + offset_unscaled,
            unscaled_areas,
            width,
            label=f"Method {i+1} Unscaled",
            alpha=0.9,
        )
        ax_area.bar(
            x + offset_scaled,
            scaled_areas,
            width,
            label=f"Method {i+1} Scaled",
            alpha=0.6,
        )
        ax_mass.bar(
            x + offset_unscaled,
            unscaled_masses,
            width,
            label=f"Method {i+1} Unscaled",
            alpha=0.9,
        )
        ax_mass.bar(
            x + offset_scaled,
            scaled_masses,
            width,
            label=f"Method {i+1} Scaled",
            alpha=0.6,
        )

    for ax, title in [(ax_area, "Area"), (ax_mass, "Mass")]:
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylim(0, 200)
        ax.axhline(100, color="r", linestyle="--", linewidth=1)
        ax.set_ylabel("Proportion of True Value (%)", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="y", labelsize=9)
        ax.legend(fontsize=9)

    fig.suptitle(main_title)

    plt.tight_layout()
    output_path = (
        Path(__file__).parent
        / "Output Files"
        / "Mask Thresholds"
        / "Comparison of Methods.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
