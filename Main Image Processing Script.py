# Packages
import matplotlib.pyplot as plt
import numpy as np

import funcs_second_order as funcs2  # Import second order functions from other file

# Turn the .tiff file into a png, save, and then a mask which is saved as a png
mask = funcs2.images_init("fake_microsope.tiff", 1300)

# Generate a list of dicts, where each dict contains info about the particles
# Also generate a recreation of the mask to verify that it all worked

recreated_mask, particle_dicts = funcs2.generate_particle_dicts(mask)

# Plot the radius distribution

data = [p["effective_radius"] for p in particle_dicts]

fig, ax = plt.subplots()
ax.hist(data, bins=500, log=True)
ax.set_xlim((np.min(data), np.max(data) + 5))
ax.set_xlabel("Particle size (pixels)")
ax.set_ylabel("Number of particles")
plt.tight_layout()
plt.show()
