import cv2
import numpy as np
import json
import funcs1
from shapely.geometry import box, Point

original_img = cv2.imread("fake_microsope 1.tiff", 0)
mask = funcs1.mask_image(original_img, 255)
_, (r0, r1, c0, c1) = funcs1.crop_to_plot_area(mask, min_length=1000)
microscope_img = original_img[r0:r1, c0:c1]

um_per_pixel = 6.5 / 60

with open("data.json") as f:
    data = json.load(f)

diameters_px = np.array(data["D"]) / um_per_pixel
radii_px = diameters_px / 2
i_max = np.argmax(radii_px)
r = radii_px[i_max]
r_i = int(round(r))

# Find actual centre from image
min_pos = np.unravel_index(np.argmin(microscope_img), microscope_img.shape)
actual_cy, actual_cx = min_pos

print(f"Largest circle actual centre: ({actual_cx}, {actual_cy}), r={r:.1f}px")

# Collect boundary pixel samples for just this circle
greyscale_vals = []
coverage_vals = []

x0 = max(0, actual_cx - r_i - 2)
x1 = min(microscope_img.shape[1], actual_cx + r_i + 2)
y0 = max(0, actual_cy - r_i - 2)
y1 = min(microscope_img.shape[0], actual_cy + r_i + 2)

cols, rows = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
dist = np.sqrt((cols - actual_cx) ** 2 + (rows - actual_cy) ** 2)
edge_mask = np.abs(dist - r) < 1.5

circle = Point(actual_cx, actual_cy).buffer(r)
edge_coords = np.argwhere(edge_mask)

for idx in edge_coords:
    py = int(rows[idx[0], idx[1]])
    px = int(cols[idx[0], idx[1]])
    pixel = box(px - 0.5, py - 0.5, px + 0.5, py + 0.5)
    true_coverage = pixel.intersection(circle).area
    greyscale = microscope_img[py, px]
    greyscale_vals.append(int(greyscale))
    coverage_vals.append(1 - true_coverage)

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(greyscale_vals, coverage_vals, alpha=0.5, s=5)
plt.xlabel("Greyscale value")
plt.ylabel("Background fraction")
plt.title("Transfer function calibration - largest particle only")
plt.show()

print(f"Greyscale range: {min(greyscale_vals)} - {max(greyscale_vals)}")
print(f"Coverage range: {min(coverage_vals):.3f} - {max(coverage_vals):.3f}")
