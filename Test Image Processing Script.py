# Packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Files
import funcs1
import funcs2  # Import second order functions from other file
import plotting

print("1")

# Define the thresholds we want and turn each into its own entry in a dict.
threshold_list = [100, 150, 200, 255]  # Out of 255

# Turn the .tiff file into a png and save it
microscope_img = cv2.imread("fake_microsope 1.tiff", 0)

# Itterate through the desired threshold scenarios and make a mask, generate
# particles and plot the distribution
method_1_dict = plotting.plot_multiple_threshold_scenarios_test(
    threshold_list, microscope_img, 255, "data.json", "Method 1"
)
method_2_1_dict = plotting.plot_multiple_threshold_scenarios_test(
    threshold_list, microscope_img, 255, "data.json", "Method 2_1"
)
method_2_2_dict = plotting.plot_multiple_threshold_scenarios_test(
    threshold_list, microscope_img, 255, "data.json", "Method 2_3"
)

# Plot a grouped bar chart of the results
plotting.grouped_bar_thresholds_methods(
    "Method 2_1 vs Method 2_2", [method_1_dict, method_2_1_dict, method_2_2_dict]
)
