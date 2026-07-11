# Packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Files
import old.funcs1 as funcs1
import old.funcs2 as funcs2  # Import second order functions from other file
import old.plotting as plotting

# Turn the .bmp file into a binary uint8 array and save it. Bitwise not since
# the image has white soiling and black background
microscope_img = cv2.bitwise_not(funcs1.bmp_to_img("01.bmp"))

# Cleanest pixel visible (given inverted image)
clean = 255 - np.min(microscope_img)
clean_w_buffer = clean - 25  # Buffer for noise. Otherwise it's one giant particle

# Define the thresholds we want and turn each into its own entry in a dict.
threshold_list = [100, 150, 200, clean_w_buffer]

plotting.plot_multiple_threshold_scenarios_real(
    threshold_list, microscope_img, clean_w_buffer, "Method 2_2"
)
