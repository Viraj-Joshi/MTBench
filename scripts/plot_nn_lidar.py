import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm # Renamed to avoid conflict if plt.cm is used later

# List of NN architectures
nn_methods = ["Multihead", "PaCO", "MH-MOORE", "MH-CARE", "Soft-Modularization"]

# Success Rate (SR) values from the table for MT10 and MT50
sr_mt10 = [86.77, 84.37, 86.94, 84.79, 82.96]
sr_mt50 = [74.03, 66.82, 79.46, 70.68, 68.51]

# Vanilla success rates for normalization
vanilla_mt10 = 87.51
vanilla_mt50 = 63.49

# Use plt.get_cmap for newer matplotlib versions
cmap = plt.get_cmap("Accent", 8)
color_mt10 = cmap(0)
color_mt50 = cmap(1)

# Compute the radius proportional to the success rate ratio with respect to Vanilla
sr_ratio_mt10 = np.array(sr_mt10) - vanilla_mt10
sr_ratio_mt50 = np.array(sr_mt50) - vanilla_mt50

# Define angles for the LiDAR plot (evenly spaced) - in radians for calculations
angles_rad = np.linspace(0, 2 * np.pi, len(nn_methods), endpoint=False)

# Close the radar chart data by appending the first value to the end for plotting lines/fills
sr_ratio_mt10_closed = np.append(sr_ratio_mt10, sr_ratio_mt10[0])
sr_ratio_mt50_closed = np.append(sr_ratio_mt50, sr_ratio_mt50[0])
angles_rad_closed = np.append(angles_rad, angles_rad[0]) # Use original angles_rad

# Create the LiDAR plot
fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection': 'polar'})

# Plot for MT10
ax.plot(angles_rad_closed, sr_ratio_mt10_closed, 'o-', linewidth=2, label="MT10", color=color_mt10)
ax.fill(angles_rad_closed, sr_ratio_mt10_closed, alpha=0.2, color=color_mt10)

# Plot for MT50
ax.plot(angles_rad_closed, sr_ratio_mt50_closed, 'o-', linewidth=2, label="MT50", color=color_mt50)
ax.fill(angles_rad_closed, sr_ratio_mt50_closed, alpha=0.2, color=color_mt50)

# Set the angular grid and labels (nn_methods)
angles_deg = angles_rad * 180 / np.pi # Convert radians to degrees for set_thetagrids
ax.set_thetagrids(angles_deg, labels=nn_methods, fontsize=8) # Removed frac argument

# Adjust the padding for the theta-axis (angular) tick labels to move them outwards
# 'axis='x'' refers to the theta-axis in polar plots.
# 'pad' is the distance in points from the tick to the label.
ax.tick_params(axis='x', which='major', pad=10) # Added padding

# Position the legend outside the plot area
# bbox_to_anchor positions the legend relative to the axes. (1,1) is upper right of axes.
# Values > 1 for x or y place it outside.
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), fontsize=8)

# Save the plot
# bbox_inches='tight' ensures that all elements (legend, labels) fit into the saved figure
plt.savefig("scripts/figures/lidar_plot.pdf", dpi=300, bbox_inches='tight')
# plt.show() # Uncomment to display the plot interactively
