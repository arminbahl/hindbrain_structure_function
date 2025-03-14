"""
------------------------------------------------------
Hindbrain Neural Data Processing and Visualization
------------------------------------------------------
Version: 0.1
Date: 10/09/2024
Author: Jonathan Boulanger-Weill

Description:
This script processes neural data from the hindbrain, including:
- Loading and visualizing regressors.
- Clustering neurons and computing neuronal activity.
- Displaying spatial and regional distributions.
- Analyzing functional types (motor command, dynamic threshold, integrators).
- Performing statistical analysis (e.g., Mann-Whitney U test).

Outputs:
- PDF visualizations for regressors and neuronal clusters.
- HDF5 data for neuron clusters and activity.

License:
This code is released under the MIT License.
------------------------------------------------------
"""
# %%
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.signal import savgol_filter 
from scipy.stats import mannwhitneyu
from collections import defaultdict, Counter
from matplotlib import rcParams
from PIL import Image
from matplotlib_scalebar.scalebar import ScaleBar
from tifffile import TiffFile
from skimage.measure import find_contours

# Set global font settings
rcParams['font.family'] = 'Arial'

PATH_OUTPUT = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/outputs'
PATH_INPUT = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/inputs'
# Load the regressors data
regressors = np.load(os.path.join(PATH_INPUT, "kmeans_regressors_final.npy"))

COLOR_CELL_TYPE_DICT = {
    "integrator_ipsilateral": (254/255, 179/255, 38/255, 0.7),      # Yellow-orange
    "integrator_contralateral": (232/255, 77/255, 138/255, 0.7),    # Magenta-pink
    "dynamic_threshold": (100/255, 197/255, 235/255, 0.7),          # Light blue
    "motor_command": (127/255, 88/255, 175/255, 0.7),               # Purple
    "raphe": (34/255, 139/255, 34/255, 0.7),                        # Forest green
    "myelinated": (255/255, 127/255, 14/255, 0.7),                  # Bright orange
    "axon_caudal": (0/255, 0/255, 0/255, 0.7),                      # Black
    "axon_rostral": (0/255, 0/255, 0/255, 0.7)                      # Black
}
# Set default font to Arial
rcParams['font.family'] = 'Arial'

##### PLOT REGRESSORS 
# %%

# Define time axis and color scheme
dt = 0.5
# Assuming regressors is already defined and is a NumPy array
regressors = np.array(regressors)  # Ensure regressors is a NumPy array
time_axis = np.arange(regressors.shape[1]) * dt

# Function to plot regressors with scale bars and axis off
def plot_regressors(ax, labels, avg_integrators=False):
    # Plot the data
    ax.plot(time_axis, regressors[0], label=labels[0], color=COLOR_CELL_TYPE_DICT['motor_command'])
    ax.plot(time_axis, regressors[1], label=labels[1], color=COLOR_CELL_TYPE_DICT['dynamic_threshold'])
    
    # If averaging integrators
    if avg_integrators:
        integrator_avg = (regressors[2] + regressors[3]) / 2
        ax.plot(time_axis, integrator_avg, label='Average Integrator', color=COLOR_CELL_TYPE_DICT['integrator_contralateral'])
    else:
        ax.plot(time_axis, regressors[2], label=labels[2], color=COLOR_CELL_TYPE_DICT['integrator_contralateral'])
        ax.plot(time_axis, regressors[3], label=labels[3], color=COLOR_CELL_TYPE_DICT['integrator_contralateral'])

    # Apply the same scale bar and styling for all subplots
    ax.axvspan(10, 50, color='gray', alpha=0.1)  # Gray span to highlight specific time range
    ax.plot([0, 10], [-0.05, -0.05], color='k', lw=2)  # 10 sec horizontal scale bar
    ax.text(5, -0.07, '10 sec', ha='center', fontfamily='Arial', fontsize=6)  # Time scale label
    ax.plot([-2, -2], [0, 0.2], color='k', lw=2)  # Vertical scale bar (0.2 a.u.)
    ax.text(-2.5, 0.1, '0.2 a.u.', va='center', fontfamily='Arial', rotation=90, fontsize=6)  # Value scale label
    
    # Hide axes to match the desired style
    ax.set_axis_off()

    return None

# Ensure the output directory exists
if not os.path.exists(PATH_OUTPUT):
    os.makedirs(PATH_OUTPUT)

# Create the figure and two subplots
width_mm = 80 / 25.4
height_mm = 40 / 25.4
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(width_mm, height_mm))

# Plot all regressors in the first subplot
plot_regressors(ax1, ['Motor command', 'Dynamic threshold', 'Integrator #1', 'Integrator #2'])

# Plot averaged integrators in the second subplot and capture the data (motor command, dynamic threshold, integrator_avg)
regressor_data = plot_regressors(ax2, ['Motor command', 'Dynamic threshold'], avg_integrators=True)

# Save concatenated data to a file when avg_integrators=True
if regressor_data is not None:
    motor_command, dynamic_threshold, integrator_avg = regressor_data
    
    # Concatenate the data along the second axis (120, 3)
    concatenated_data = np.column_stack((motor_command, dynamic_threshold, integrator_avg))
    
    # Check shape for confirmation
    print(f"Concatenated data shape: {concatenated_data.shape}")  # Should be (120, 3)
    
    # Save the concatenated data
    output_file = os.path.join(PATH_OUTPUT, 'kmeans_regressors_280824_concat.npy')
    np.save(output_file, concatenated_data)
    print(f"Concatenated regressors saved to {output_file}")

# Add legends at the bottom of each subplot
for ax in [ax1, ax2]:
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, prop={'family': 'Arial', 'size': 6})

plt.tight_layout(rect=[0, 0.1, 1, 1])

# Save the figure
file_name = 'regressors_final.pdf'
file_path = os.path.join(PATH_OUTPUT, file_name)
plt.savefig(file_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Figure saved to {file_path}")
plt.show()

##### CLUSTER NEURONS 
# %%
regressors = np.load('/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/outputs/kmeans_regressors_280824_concat.npy')
plt.plot(regressors[:,0])
plt.plot(regressors[:,1])
plt.plot(regressors[:,2])
plt.show()
print(regressors.shape)

container_functional_hdf5 = h5py.File(os.path.join(PATH_INPUT,'functional_2019-10-29_15-12-39_all_data.hdf5'), "r", libver='latest', swmr=True)
                           
# Initialize a neuron counter
neuron_counter = 0

# Create the HDF5 file
with h5py.File("/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/all_cells_091024.h5", "w") as hdf_file:
    for z_plane in range(6):

        segmentation_hdf5 = container_functional_hdf5[f"{z_plane}/manual_segmentation"]
        number_of_units = segmentation_hdf5.attrs["number_of_units"]
        dt = segmentation_hdf5["stimulus_aligned_dynamics"].attrs["dt"]

        print(f"Processing z_plane {z_plane}, number of units: {number_of_units}")

        F_left_dots = np.array(segmentation_hdf5[f"stimulus_aligned_dynamics/0/F"])
        F_right_dots = np.array(segmentation_hdf5[f"stimulus_aligned_dynamics/1/F"])

        # Compute deltaF/F for each trial, for the 4 tested stimuli
        F0_left_dots = np.nanmean(F_left_dots[:, :, int(10 / dt):int(20 / dt)], axis=2, keepdims=True)
        df_F_left_dots = 100 * (F_left_dots - F0_left_dots) / F0_left_dots

        F0_right_dots = np.nanmean(F_right_dots[:, :, int(10 / dt):int(20 / dt)], axis=2, keepdims=True)
        df_F_right_dots = 100 * (F_right_dots - F0_right_dots) / F0_right_dots

        # Add the centroids positions 
        unit_centroids = segmentation_hdf5["unit_centroids"]
        # The cells have also been registered to the z-brain, get those coordinates as well
        unit_centroids_ants_registered = segmentation_hdf5["unit_centroids_ants_registered"]

        # Loop through each neuron
        cluster_IDs = []
        for neuron_idx in range(number_of_units):

            #The first neuron is 1 not 0, 0 is the background in segmented planes
            neuron_counter += 1

            # Activity for each neuron
            neuron_df_F_left_dots = df_F_left_dots[:, neuron_idx, :]
            neuron_df_F_right_dots = df_F_right_dots[:, neuron_idx, :]

            # Compute average neuronal activity for each neuron and each stimulus type
            avg_df_F_left_dots = np.nanmean(neuron_df_F_left_dots, axis=0)
            avg_df_F_right_dots = np.nanmean(neuron_df_F_right_dots, axis=0)
            
            # Create groups for each neuron
            neuron_group = hdf_file.create_group(f"neuron_{neuron_counter}")
            # Store average activity for left and right stimuli
            neuron_group.create_dataset("average_activity_left", data=avg_df_F_left_dots)
            neuron_group.create_dataset("average_activity_right", data=avg_df_F_right_dots)
            # Store individual trial data for left and right stimuli
            neuron_group.create_dataset("neuronal_activity_trials_left", data=neuron_df_F_left_dots)
            neuron_group.create_dataset("neuronal_activity_trials_right", data=neuron_df_F_right_dots)

            # As we have the cell registered to the z-brain, we know if it is on the left or right hemisphere
            x_ants_registered = unit_centroids_ants_registered[neuron_idx, 0]
            neuron_group.create_dataset("x_position_reg", data=x_ants_registered)
            if x_ants_registered < 314:
                pref_dir='left'
            else:
                pref_dir='right'

            ########################
            # Compute time constant 
            if pref_dir == 'left':
                PD = avg_df_F_left_dots
            elif pref_dir == 'right':
                PD = avg_df_F_right_dots
            peak = 0.90 * np.nanmax(PD[40:120])
            peak_indices = np.where(PD[40:120] >= peak)
            
            # Check if peak_indices has any values, otherwise assign None or a default value
            if peak_indices[0].size > 0:
                time_constant_value = peak_indices[0][0] / 2
                neuron_group.create_dataset("time_constant", data=time_constant_value)
            else: 
                neuron_group.create_dataset("time_constant", data=np.nan)
       
            # Compute reliability
            if pref_dir == 'left':
                st = neuron_df_F_left_dots[:,40:120]
            elif pref_dir == 'right':   
                st = neuron_df_F_right_dots[:,40:120]
            rel = np.round(np.nanmean(np.mean(st, axis=0) / np.nanstd(st, axis=0), axis=0),2)
            neuron_group.create_dataset("reliability", data=rel)

            # Compute direction selectivity 
            if pref_dir == 'left': 
                PD = avg_df_F_left_dots
                ND = avg_df_F_right_dots    
                maxresp_L= np.nanmax(PD[40:120])
                if maxresp_L < 0: 
                    maxresp_L = 0
                maxresp_R= np.nanmax(ND[40:120])
                if maxresp_R < 0: 
                    maxresp_R = 0
            elif pref_dir == 'right':
                PD = avg_df_F_right_dots
                ND = avg_df_F_left_dots 
                maxresp_R= np.nanmax(PD[40:120])
                if maxresp_R < 0: 
                    maxresp_R = 0
                maxresp_L= np.nanmax(ND[40:120]) ; 
                if maxresp_L < 0: 
                    maxresp_L = 0
            if maxresp_R == 0 and maxresp_L == 0: 
                ds = np.nan  
            else: 
                ds = (maxresp_R - maxresp_L) / (maxresp_R + maxresp_L)
            neuron_group.create_dataset("direction_selectivity", data=ds)

            ########################
            # Cluster neurons 
            # We drop the first and last 10 s, as this is how the regressors had been computed
            if x_ants_registered < 314:
                PD = avg_df_F_left_dots[int(10/dt):int(70/dt)] 
                ND = avg_df_F_right_dots[int(10/dt):int(70/dt)]
                pref_dir='left'
            else:
                PD = avg_df_F_right_dots[int(10/dt):int(70/dt)]
                ND = avg_df_F_left_dots[int(10/dt):int(70/dt)]
                pref_dir='right'

            # Compute the correleation coefficient to all three regressors
            ci = [np.corrcoef(PD, regressors[:,0])[0, 1],
                np.corrcoef(PD, regressors[:,1])[0, 1],
                np.corrcoef(PD, regressors[:,2])[0, 1]]

            # Which one has the highest correlation
            i_max = np.argmax(ci)

            # if the highest correlation is above a certain threshold and the maximal response of the cell is more than 20% df/F, assign cell type
            if ci[i_max] > 0.80 and np.abs(PD[10:].max()) > 20:
                neuron_group.create_dataset("cluster_id", data=i_max)
            else:
                neuron_group.create_dataset("cluster_id", data=-1)
            
            # Store centroids positions 
            neuron_unit_centroids=unit_centroids[neuron_idx, :]
            neuron_group.create_dataset("neuron_positions", data=np.append(neuron_unit_centroids, z_plane))    
neuron_counter

##### DISPLAY SPATIAL DISTRIBUTIONS 
# %%

# File paths
img_path = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/inputs/AVG_functional_stack_green_gauss35_gamma70.png'
hdf5_path = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/all_cells_091024.h5"

# Load the background image
img = Image.open(img_path)

# Color mapping for clusters
cluster_to_color = {
    0: COLOR_CELL_TYPE_DICT["motor_command"],          # Red (MC)
    1: COLOR_CELL_TYPE_DICT["dynamic_threshold"],      # Blue (DT)
    2: COLOR_CELL_TYPE_DICT["integrator_contralateral"] # Magenta (INT)
}

# Extract neuron data: positions, colors, and clusters
neuron_positions = []
neuron_colors = []
neuron_clusters = []  # For counting clusters

with h5py.File(hdf5_path, "r") as hdf_file:
    for neuron_idx in range(1, 15018): 
        neuron_group = hdf_file[f"neuron_{neuron_idx}"]
        cluster = neuron_group["cluster_id"][()]

        # Handle positions and colors for scatter plot
        if cluster in [0, 1, 2]:  # Only plot valid clusters
            position = neuron_group["neuron_positions"][:]
            neuron_positions.append(position)
            neuron_colors.append(cluster_to_color.get(cluster, (0, 0, 0, 0)))  # Default transparent if not found
            neuron_clusters.append(cluster)  # Track cluster for bar plot

# Define the size of the figure
width_mm = 40 / 25.4  # Convert millimeters to inches
height_mm = 40 / 25.4
fig, ax = plt.subplots(figsize=(width_mm, height_mm))

# Plot the background image
ax.imshow(img, cmap='gray')

# Scatter plot of neurons
for position, color in zip(neuron_positions, neuron_colors):
    ax.scatter(position[0], position[1], color=color, s=0.5)
ax.axis('off')
scalebar = ScaleBar(0.48, units="µm", location='lower right', frameon=False, color='white', box_color='black')
ax.add_artist(scalebar)

# Save the figure
file_name = 'scatter_clusters_test.pdf'
file_path = os.path.join(PATH_OUTPUT, file_name)
plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0, format='pdf')

print(f"Figure saved to {file_path}")
plt.show()

# Create a bar plot showing neuron counts for clusters 0, 1, and 2
cluster_labels = ["INT", "DT", "MC"]  # Labels: INT (Integrator), DT (Dynamic Threshold), MC (Motor Command)
cluster_indices = [2, 1, 0]           # Corresponding cluster order
cluster_counts = Counter(neuron_clusters)
bar_colors = [COLOR_CELL_TYPE_DICT["integrator_contralateral"], 
              COLOR_CELL_TYPE_DICT["dynamic_threshold"], 
              COLOR_CELL_TYPE_DICT["motor_command"]]

width_mm = 30 / 25.4
height_mm = 40 / 25.4
plt.figure(figsize=(width_mm, height_mm))
bars = plt.bar(cluster_labels, [cluster_counts[i] for i in cluster_indices], color=bar_colors, width=0.5, edgecolor='black')

# Add value labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 20, int(yval), ha='center', va='bottom', fontsize=6)

# Format the bar plot
plt.ylabel('Number of neurons', fontsize=6)
plt.xlabel('Functional types', fontsize=6)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.ylim(0, 700)
plt.tight_layout()

# Show the final bar plot
file_name = 'bar_clusters_final.pdf'
file_path = os.path.join(PATH_OUTPUT, file_name)
plt.savefig(file_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Figure saved to {file_path}")
plt.show()

##### DISPLAY REGIONAL DISTRIBUTIONS 
# %%

# Path to your CSV file
path_regions = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/inputs/regions_120621.csv'
brain_reg_temp = pd.read_csv(path_regions)

# Define regions to analyze
regions_to_analyze = ['Midbrain', 'Pretectum', 'Cerebellum', 'Hindbrain,Rhombomere 1', 'Hindbrain,Rhombomere 2', 'Hindbrain,Rhombomere 3']

# Group 'Hindbrain,Rhombomere 1' and 'Hindbrain,Rhombomere 2'
region_mapping = {
    'Hindbrain,Rhombomere 1': 'Hindbrain,Rhombomere 1-3',
    'Hindbrain,Rhombomere 2': 'Hindbrain,Rhombomere 1-3', 
    'Hindbrain,Rhombomere 3': 'Hindbrain,Rhombomere 1-3'
}

def map_region(region_str):
    if pd.isna(region_str):
        return None
    region_str = str(region_str)
    for region, grouped_region in region_mapping.items():
        if region in region_str:
            return grouped_region
    for region in regions_to_analyze:
        if region in region_str:
            return region
    return None

# Apply the region mapping
brain_reg_temp['region_grouped'] = brain_reg_temp.iloc[:, 3].apply(map_region)

# Initialize a dictionary to count neurons by region and cluster
neuron_counts = {region: defaultdict(int) for region in regions_to_analyze}
neuron_counts['Hindbrain,Rhombomere 1-3'] = defaultdict(int)
neuron_counts['Other'] = defaultdict(int)

# Extract neuron data from the HDF5 file
hdf5_path = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/all_cells_091024.h5"
with h5py.File(hdf5_path, "r") as hdf_file:
    for neuron_idx in range(1, 15018):  # Range from 1 to 15017 inclusive
        neuron_name = f"neuron_{neuron_idx}"
        if neuron_name in hdf_file:
            neuron_group = hdf_file[neuron_name]
            cluster = neuron_group["cluster_id"][()]

            if cluster == -1:
                continue

            region = brain_reg_temp.loc[neuron_idx - 1, 'region_grouped']
            if region in neuron_counts:
                neuron_counts[region][cluster] += 1
            else:
                neuron_counts['Other'][cluster] += 1

# Prepare data for plotting
regions = ['Midbrain', 'Pretectum', 'Cerebellum', 'Hindbrain,Rhombomere 1-3', 'Other']
clusters = [0, 1, 2]
counts = {region: [neuron_counts[region][cluster] for cluster in clusters] for region in regions}
bar_width = 0.5

# Plot the stacked histogram
width_mm = 40 / 25.4
height_mm = 40 / 25.4
fig, ax = plt.subplots(figsize=(width_mm, height_mm))
bar_positions = range(len(regions))

# Plot bars for each cluster
bottom_counts = [0] * len(regions)
for cluster in clusters:
    cluster_counts = [counts[region][cluster] for region in regions]
    ax.bar(bar_positions, cluster_counts, bottom=bottom_counts, 
           color=cluster_to_color[cluster], width=bar_width, 
           edgecolor='black', label=f'Cluster {cluster}')

    bottom_counts = [bottom + count for bottom, count in zip(bottom_counts, cluster_counts)]

# Formatting the plot to match the requested design
ax.set_xlabel('Regions', fontsize=6)
ax.set_ylabel('Number of neurons', fontsize=6)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
ax.set_xticks(bar_positions)
regions_short_names = ['Mb', 'Pt', 'Cb', 'Hb, Rom1-3', 'Other']
ax.set_xticklabels(regions_short_names, fontsize=6)
plt.ylim(0, 700)

# Finalize the plot aesthetics
plt.tight_layout()

file_name = 'regional_distributions_final.pdf'
file_path = os.path.join(PATH_OUTPUT, file_name)
# plt.savefig(file_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Figure saved to {file_path}")
plt.show()

##### DISPLAY SINGLE PLANE CLUSTERING
# %%

# Path settings
hdf5_file_path = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/all_cells_091024.h5"
masks_path = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/inputs/2d_masks_sd_500e/'
imgs_path = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/inputs/2d_images/'

# Color dictionary for neuron clusters
COLOR_CELL_TYPE_DICT_LINES = {
    "integrator_contralateral": (232/255, 77/255, 138/255, 1),    # Magenta-pink
    "dynamic_threshold": (100/255, 197/255, 235/255, 1),          # Light blue
    "motor_command": (127/255, 88/255, 175/255, 1),               # Purple
    "default": (0.5, 0.5, 0.5, 0.7)                                # Gray for undefined clusters
}
size_line = 0.7
max_windowWidth = 11  # Maximum window size for Savitzky-Golay filter
polynomialOrder = 3  # Polynomial order for the filter

# Load the HDF5 file (neuron positions and cluster IDs)
with h5py.File(hdf5_file_path, 'r') as hdf_file:
    neuron_positions = []
    cluster_ids = []
    for neuron_id in hdf_file.keys():
        neuron_group = hdf_file[neuron_id]
        positions = neuron_group['neuron_positions'][()]
        neuron_positions.append(positions[:3])  # Store x, y, z coordinates
        cluster_idx = neuron_group["cluster_id"][()]
        cluster_ids.append(cluster_idx)

# Convert neuron positions and cluster IDs to NumPy arrays for easier indexing
neuron_positions = np.array(neuron_positions)
cluster_ids = np.array(cluster_ids)

# Loop through each z-plane (from 0 to 5)
for z_plane in range(6):
    print(f"Processing neurons in z-plane {z_plane}")

    # Load the corresponding segmentation and background files for the current z-plane
    segmentation_file_path = os.path.join(masks_path, f'fp_gau35_ga70_{z_plane}_sd_masks.tif')
    img_file_path = os.path.join(imgs_path, f'fp_gau35_ga70_{z_plane}.tif')
    
    # Load the TIFF files
    with TiffFile(segmentation_file_path) as tif:
        patch_data = tif.asarray()
    with TiffFile(img_file_path) as tif:
        background_img = tif.asarray()

    # Filter neurons in the current z-plane
    neurons_in_plane = neuron_positions[neuron_positions[:, 2] == z_plane]
    cluster_ids_in_plane = cluster_ids[neuron_positions[:, 2] == z_plane]

    # If no neurons, skip to the next plane
    if len(neurons_in_plane) == 0:
        print(f"No neurons in z-plane {z_plane}")
        continue

    # Create the plot with high DPI for better resolution
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)  # Set figure size and DPI

    # Improve the contrast of the background image
    ax.imshow(background_img, cmap='gray', interpolation='none')

    # Loop through each neuron in the current z-plane
    for neuron_idx, position in enumerate(neurons_in_plane):
        x, y, z = position  # Extract x, y, z coordinates
        x, y = int(x), int(y)  # Convert to integer indices for indexing
        
        # Find the patch value at the neuron's (x, y) position
        patch_value = patch_data[y, x]

        # Create a binary mask for the current patch
        patch_mask = (patch_data == patch_value)

        # Use find_contours to get the perimeter of the patch
        contours = find_contours(patch_mask, level=0.5)

        # Plot each contour
        for contour in contours:
            # Dynamically adjust window size based on contour length
            contour_length = len(contour[:, 1])
            windowWidth = min(max_windowWidth, contour_length - 1) if contour_length > 1 else 1

            # Smooth the contour using Savitzky-Golay filter if window size is valid
            if windowWidth > 2:
                smoothX = savgol_filter(contour[:, 1], windowWidth, polynomialOrder)
                smoothY = savgol_filter(contour[:, 0], windowWidth, polynomialOrder)
            else:
                smoothX, smoothY = contour[:, 1], contour[:, 0]  # If contour too small, skip smoothing

            # Determine the cluster ID and corresponding color
            cluster_idx = cluster_ids_in_plane[neuron_idx]
            if cluster_idx == 0:
                color = COLOR_CELL_TYPE_DICT_LINES["motor_command"]
            elif cluster_idx == 1:
                color = COLOR_CELL_TYPE_DICT_LINES["dynamic_threshold"]
            elif cluster_idx == 2:
                color = COLOR_CELL_TYPE_DICT_LINES["integrator_contralateral"]
            else:
                color = COLOR_CELL_TYPE_DICT_LINES["default"]

            # Plot the smoothed boundaries with the corresponding color
            ax.plot(smoothX, smoothY, color=color, linewidth=size_line)

        # Print the neuron index (for tracking progress)
        print(f"Neuron {neuron_idx} at z-plane {z_plane}")

    # Add a scale bar (pixel size = 0.48 μm)
    scalebar = ScaleBar(0.48, units="µm", location='lower right', frameon=False, color='white', box_color='black')
    ax.add_artist(scalebar)

    # Save the final image with overlays as a PDF
    output_pdf_path = os.path.join(PATH_OUTPUT, f'neuron_boundaries_z_plane_{z_plane}.pdf')
    output_png_path = os.path.join(PATH_OUTPUT, f'neuron_boundaries_z_plane_{z_plane}.png')
    ax.axis('off')  # Turn off the axes
    plt.savefig(output_pdf_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0) 
    plt.savefig(output_png_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0) 
    print(f"Saved output as {output_pdf_path}")
    plt.close(fig)

# Crop a little snippet on the 3rd plane shoing the segmentation
# %%
def crop_png(input_png_path, output_png_path, crop_area):
    """
    Crop a specific area from a PNG image and save it as a new PNG.
    
    :param input_png_path: Path to the input PNG file
    :param output_png_path: Path to save the cropped PNG file
    :param crop_area: Tuple (x0, y0, x1, y1) defining the crop area in pixels
    """
    # Open the PNG image
    img = Image.open(input_png_path)
    
    # Crop the image using the defined crop area (x0, y0, x1, y1)
    cropped_img = img.crop(crop_area)
    
    # Save the cropped image as a new PNG file
    cropped_img.save(output_png_path)
    print(f"Cropped PNG saved to {output_png_path}")

# Example usage
input_png_path = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/outputs/neuron_boundaries_z_plane_3.png'
output_png_path = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/outputs/neuron_boundaries_z_plane_3_cropped.png'
crop_area = (483, 972, 483+280, 972+280)  # Define the crop area in pixels (x0, y0, x1, y1)

# Crop the PNG image
crop_png(input_png_path, output_png_path, crop_area)

##### DISPLAY ACTIVITY OF KMEANS CLUSTERED CELLS 
# %%
path_all_cells='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_240924.xlsx'
df = pd.read_excel(path_all_cells)

# First extract IDs of all neuron types 
int_ipsi_ids = df[(df.iloc[:, 9] == 'integrator') & (df.iloc[:, 11].str.contains('ipsilateral'))].iloc[:,1]
int_contra_ids = df[(df.iloc[:, 9] == 'integrator') & (df.iloc[:, 11].str.contains('contralateral'))].iloc[:,1]
dt_ids = df[(df.iloc[:, 9] == 'dynamic threshold')].iloc[:,1]
mc_ids = df[(df.iloc[:, 9] == 'motor command')].iloc[:,1]
def plot_max_activity(cell_ids, hdf5_file_path, color_cell_type_dict_norm, cell_type, out_dir, dt=0.5):
    with h5py.File(hdf5_file_path, "r") as hdf_file:
        fig, ax = plt.subplots()

        # Initialize an array to accumulate the max activity traces
        all_max_arrays = []
        rise_time_constants = []

        for idx in cell_ids:
            neuron_group = hdf_file[f"neuron_{idx}"]
            avg_activity_left = neuron_group["average_activity_left"][()]
            avg_activity_right = neuron_group["average_activity_right"][()]
            smooth_avg_activity_left = savgol_filter(avg_activity_left, 20, 3)
            smooth_avg_activity_right = savgol_filter(avg_activity_right, 20, 3)

            # Find the maximum values in each array
            max_left = np.max(smooth_avg_activity_left[40:120])
            max_right = np.max(smooth_avg_activity_right[40:120])

            # Determine which array has the maximum value
            if max_left > max_right:
                max_array = smooth_avg_activity_left
            else:
                max_array = smooth_avg_activity_right

            # Normalize activity 
            max_array = (max_array / np.nanmax(max_array)) * 100
            # Accumulate the max activity traces
            all_max_arrays.append(max_array)

            # Compute rise time constant from the stim onset 
            peak = 0.90 * np.nanmax(max_array[40:120])  # 90% of the peak
            peak_indices = np.where(max_array >= peak)[0]
            if peak_indices.size > 0 & peak_indices.size < 120:
                peak_index = peak_indices[0]
                rise_time_constant = (peak_index - 40) * dt
            else:
                rise_time_constant = np.nan  # Assign NaN if indices are not found
                print(f"Could not find peak for neuron {idx}")

            rise_time_constants.append(rise_time_constant)

            # Define time axis in seconds
            time_axis = np.arange(len(avg_activity_left)) * dt

            plt.plot(time_axis, max_array, color=color_cell_type_dict_norm.get(cell_type), alpha=0.7, linestyle='-', linewidth=1)

        # Plot the mean of max_responses
        mean_max_array = np.nanmean(all_max_arrays, axis=0)
        plt.plot(time_axis, mean_max_array, color='black', alpha=0.7, linestyle='-', linewidth=2)

        # Overlay shaded rectangle for stimulus epoch
        plt.axvspan(20, 60, color='gray', alpha=0.1)

        # Remove the axes and add the scale bars
        ax.plot([0, 10], [-5, -5], color='k', lw=2)  # Time scale bar (10 sec)
        ax.text(5, -7, '10 sec', ha='center', fontfamily='Arial', fontsize=14)

        # Adapted scale bar for normalized activity (using 10% of the normalized scale)
        ax.plot([-2, -2], [0, 10], color='k', lw=2)  # Activity scale bar (10% of normalized activity)
        ax.text(-2.5, 5, '10%', va='center', fontfamily='Arial', rotation=90, fontsize=14)

        # Set aspect ratio to 1 and remove the axis lines
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)))
        ax.set_axis_off()  # Remove the axis

        plt.show()

        # Save the figure
        filename = f"{cell_type}_activity-traces.pdf"
        output_path = os.path.join(out_dir, filename)
        fig.savefig(output_path, dpi=1200)
        print(f"Figure saved successfully at: {output_path}")

        return rise_time_constants
    
rise_tc_int_ipsi = plot_max_activity(int_ipsi_ids, hdf5_file_path, COLOR_CELL_TYPE_DICT, "integrator_ipsilateral", PATH_OUTPUT, dt=0.5)
rise_tc_int_contra = plot_max_activity(int_contra_ids, hdf5_file_path, COLOR_CELL_TYPE_DICT, "integrator_contralateral", PATH_OUTPUT, dt=0.5)
plot_max_activity(dt_ids, hdf5_file_path, COLOR_CELL_TYPE_DICT, "dynamic_threshold", PATH_OUTPUT, dt=0.5)
plot_max_activity(mc_ids, hdf5_file_path, COLOR_CELL_TYPE_DICT, "motor_command", PATH_OUTPUT, dt=0.5)

## Time constant analysis 
rise_tc_int_ipsi = np.array(rise_tc_int_ipsi)
rise_tc_int_ipsi = rise_tc_int_ipsi[~np.isnan(rise_tc_int_ipsi)]
rise_tc_int_contra = np.array(rise_tc_int_contra)
rise_tc_int_contra = rise_tc_int_contra[~np.isnan(rise_tc_int_contra)]
u_statistic, p_value = mannwhitneyu(rise_tc_int_ipsi, rise_tc_int_contra, alternative='less')
print(f"Mann-Whitney U test: U statistic = {u_statistic}, p-value = {p_value}")

# Bar Plot with Overlayed Points
fig, axs = plt.subplots()  # 2 rows, 2 columns for square subplots
mean_fp = np.mean(rise_tc_int_ipsi)
mean_tp = np.mean(rise_tc_int_contra)
bar_positions = np.arange(2)
colors=[(0.996078431372549,0.7019607843137254, 0.14901960784313725, 1), (0.9098039215686274, 0.30196078431372547, 0.5411764705882353, 1)]
bars = axs.bar(bar_positions, [mean_fp, mean_tp], width=0.4, color=colors, edgecolor='black', linewidth=2)
jitter_strength = 0.07
jittered_positions_fp = np.random.normal(bar_positions[0], jitter_strength, size=len(rise_tc_int_ipsi))
jittered_positions_tp = np.random.normal(bar_positions[1], jitter_strength, size=len(rise_tc_int_contra))
axs.scatter(jittered_positions_fp, rise_tc_int_ipsi, color='black', s=20, zorder=3, alpha=0.7)  # smaller size with s=20
axs.scatter(jittered_positions_tp, rise_tc_int_contra, color='black', s=20, zorder=3, alpha=0.7)  # smaller size with s=20
axs.set_ylabel('Time constant (sec)', fontsize=12)
axs.set_xticks(bar_positions)
axs.set_xticklabels(['Ipsilateral integrator', 'Contralateral integrator'], fontsize=12)
axs.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%

##### FUNCTIONAL PROPERTIES
# For all Neurons 

# Extract neuron data: positions, colors, and clusters
neuron_positions = []
neuron_colors = []
neuron_clusters = []  # For counting clusters
neuron_ds = []
neuron_rel = []
neuron_tc = []

with h5py.File(hdf5_path, "r") as hdf_file:
    for neuron_idx in range(1, 15018): 
        neuron_group = hdf_file[f"neuron_{neuron_idx}"]
        cluster = neuron_group["cluster_id"][()]
        direction_selectivity = neuron_group["direction_selectivity"][()]
        reliability = neuron_group["reliability"][()]
        time_constant = neuron_group["time_constant"][()]
        
        # Handle positions and colors for scatter plot
        if cluster in [0, 1, 2]:  # Only plot valid clusters
            position = neuron_group["neuron_positions"][:]
            neuron_positions.append(position)
            neuron_colors.append(cluster_to_color.get(cluster, (0, 0, 0, 0)))  # Default transparent if not found
            neuron_clusters.append(cluster)  # Track cluster for bar plot
            neuron_ds.append(direction_selectivity)
            neuron_rel.append(reliability)
            neuron_tc.append(time_constant)

##### DIRECTION SELECTIVITY ANALYSES
# Set figure dimensions
width_mm = 60 / 25.4  # Convert mm to inches
height_mm = 60 / 25.4

# Create a 1x2 subplot layout
fig, axs = plt.subplots(1, 2, figsize=(width_mm * 2, height_mm))

# Scatter plot with background image (left side)
axs[0].imshow(img, cmap='gray')
scatter = axs[0].scatter(neuron_positions_2d[:, 0], neuron_positions_2d[:, 1], 
                         c=neuron_ds, s=0.5, cmap='coolwarm', vmin=-1, vmax=1)
#axs[0].set_title('Neuron Direction Selectivity')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_aspect('equal')
axs[0].axis('off')  # Removes the axes for a similar style

# Add scale bar
scalebar = ScaleBar(0.48, units="µm", location='lower right', frameon=False, color='white', box_color='black')
axs[0].add_artist(scalebar)

# Color bar for scatter plot
cbar = fig.colorbar(scatter, ax=axs[0], orientation='vertical', label='Direction selectivity')
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

# Histogram (right side)
axs[1].hist(neuron_ds, bins=20, color='skyblue', edgecolor='black')
#axs[1].set_title('Direction Selectivity Distribution')
#axs[1].set_xlabel('Direction selectivity')
#axs[1].set_ylabel('Number of neurons')

# Display the plot
plt.tight_layout()
file_name = 'direction_selectivity.pdf'
file_path = os.path.join(PATH_OUTPUT, file_name)
plt.savefig(file_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Figure saved to {file_path}")
plt.show()

# How many neurons had an abs(DS)>0.5
count = np.sum(np.abs(neuron_ds) > 0.5)
prct_high_ds= count/len(neuron_ds)*100

##### RELIABILITY ANALYSES
# For all clustered types
neuron_rel_clip = np.clip(neuron_rel, 0, 3)  # Clip to avoid negative values (Log scale requires positive values)

# Create a 1x2 subplot layout
fig, axs = plt.subplots(1, 2, figsize=(width_mm * 2, height_mm))

# Scatter plot with background image (left side)
axs[0].imshow(img, cmap='gray')
scatter = axs[0].scatter(neuron_positions_2d[:, 0], neuron_positions_2d[:, 1], 
                         c=neuron_rel_clip, s=0.5, cmap='viridis', vmin=0, vmax=3)
#axs[0].set_title('')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_aspect('equal')
axs[0].axis('off')  # Removes the axes for a similar style

# Add scale bar
scalebar = ScaleBar(0.48, units="µm", location='lower right', frameon=False, color='white', box_color='black')
axs[0].add_artist(scalebar)

# Color bar for scatter plot
cbar = fig.colorbar(scatter, ax=axs[0], orientation='vertical', label='Reliability')
cbar.set_ticks([0, 1, 2, 3])

# Histogram (right side)
axs[1].hist(neuron_rel_clip, bins=20, color='skyblue', edgecolor='black')
axs[1].set_xlim([0, 3])
#axs[1].set_title('')
#axs[1].set_xlabel('')
#axs[1].set_ylabel('Number of neurons')

# Display the plot
plt.tight_layout()
file_name = 'reliability.pdf'
file_path = os.path.join(PATH_OUTPUT, file_name)
plt.savefig(file_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Figure saved to {file_path}")
plt.show()

##### TIME CONSTANT  ANALYSES
# For integrator neurons, seprated by regions: Pt, Mb, Hb. 

# Create a 1x2 subplot layout
fig, axs = plt.subplots(1, 2, figsize=(width_mm * 2, height_mm))

# Scatter plot with background image (left side)
axs[0].imshow(img, cmap='gray')
scatter = axs[0].scatter(neuron_positions_2d[:, 0], neuron_positions_2d[:, 1], 
                         c=neuron_tc, s=0.5, cmap='coolwarm', vmin=0, vmax=40)
#axs[0].set_title('Neuron Direction Selectivity')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_aspect('equal')
axs[0].axis('off')  # Removes the axes for a similar style

# Add scale bar
scalebar = ScaleBar(0.48, units="µm", location='lower right', frameon=False, color='white', box_color='black')
axs[0].add_artist(scalebar)

# Color bar for scatter plot
cbar = fig.colorbar(scatter, ax=axs[0], orientation='vertical', label='Time constant')
cbar.set_ticks([0, 10, 20, 30, 40])

# Histogram (right side)
axs[1].hist(neuron_tc, bins=20, color='skyblue', edgecolor='black')
#axs[1].set_title('')
#axs[1].set_xlabel('')
#axs[1].set_ylabel('')

# Display the plot
plt.tight_layout()
file_name = 'time_constants.pdf'
file_path = os.path.join(PATH_OUTPUT, file_name)
plt.savefig(file_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Figure saved to {file_path}")
plt.show()


##### REGIONAL STATISTICS
import h5py
import pandas as pd
from collections import defaultdict

# Path to your CSV file
path_regions = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/inputs/regions_120621.csv'
brain_reg_temp = pd.read_csv(path_regions)

# Define regions to analyze and mapping for grouping specific regions
regions_to_analyze = ['Midbrain', 'Pretectum', 'Cerebellum', 'Hindbrain,Rhombomere 1', 'Hindbrain,Rhombomere 2', 'Hindbrain,Rhombomere 3']
region_mapping = {
    'Hindbrain,Rhombomere 1': 'Hindbrain,Rhombomere 1-3',
    'Hindbrain,Rhombomere 2': 'Hindbrain,Rhombomere 1-3',
    'Hindbrain,Rhombomere 3': 'Hindbrain,Rhombomere 1-3'
}

# Function to map regions according to the mapping defined above
def map_region(region_str):
    if pd.isna(region_str):
        return None
    region_str = str(region_str)
    for region, grouped_region in region_mapping.items():
        if region in region_str:
            return grouped_region
    for region in regions_to_analyze:
        if region in region_str:
            return region
    return 'Other'

# Apply the region mapping
brain_reg_temp['region_grouped'] = brain_reg_temp.iloc[:, 3].apply(map_region)

# Initialize structures to store neuron data
neuron_counts = []
neuron_counts = {region: {'count': defaultdict(int), 'indices': defaultdict(list)} for region in regions_to_analyze}
neuron_counts['Hindbrain,Rhombomere 1-3'] = {'count': defaultdict(int), 'indices': defaultdict(list)}
neuron_counts['Other'] = {'count': defaultdict(int), 'indices': defaultdict(list)}

neuron_positions = []
neuron_colors = []
neuron_clusters = []
neuron_ds = []
neuron_rel = []
neuron_tc = []
idx_clustered = 0 

# Path to the HDF5 file with neuron data
hdf5_path = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/all_cells_091024.h5"
with h5py.File(hdf5_path, "r") as hdf_file:
    for neuron_idx in range(1, 15018):  # Range from 1 to 15017 inclusive
        neuron_name = f"neuron_{neuron_idx}"
        if neuron_name in hdf_file:
            neuron_group = hdf_file[neuron_name]
            cluster = neuron_group["cluster_id"][()]
            direction_selectivity = neuron_group["direction_selectivity"][()]
            reliability = neuron_group["reliability"][()]
            time_constant = neuron_group["time_constant"][()]
            
            # Map region based on the provided CSV data
            region = brain_reg_temp.loc[neuron_idx - 1, 'region_grouped']
            if cluster == -1 or region is None:
                continue  # Skip invalid clusters or unmapped regions
            
            # Only process if the cluster is in [0, 1, 2]
            if cluster in [0, 1, 2]:
                
                # Add neuron data to the respective lists for plotting and analysis
                position = neuron_group["neuron_positions"][:]
                neuron_positions.append(position)
                neuron_clusters.append(cluster)  # Track cluster for bar plot
                neuron_ds.append(direction_selectivity)
                neuron_rel.append(reliability)
                neuron_tc.append(time_constant)

                # Increment the count and add the index to the list for the corresponding cluster and region
                if region in neuron_counts:
                    neuron_counts[region]['count'][cluster] += 1
                    neuron_counts[region]['indices'][cluster].append(idx_clustered)
                else:
                    neuron_counts['Other']['count'][cluster] += 1
                    neuron_counts['Other']['indices'][cluster].append(idx_clustered)

                idx_clustered=idx_clustered+1

##### MAKE FIGURES
hindbrain_indices = (
    neuron_counts['Hindbrain,Rhombomere 1-3']['indices'][0] +
    neuron_counts['Hindbrain,Rhombomere 1-3']['indices'][1] +
    neuron_counts['Hindbrain,Rhombomere 1-3']['indices'][2]
)
pretectum_indices= (
    neuron_counts['Pretectum']['indices'][0] +
    neuron_counts['Pretectum']['indices'][1] +
    neuron_counts['Pretectum']['indices'][2]
)
midbrain_indices= (
    neuron_counts['Midbrain']['indices'][0] +
    neuron_counts['Midbrain']['indices'][1] +
    neuron_counts['Midbrain']['indices'][2]
)

#Direction Selectivity
pt_ds = np.nanmean(np.abs([neuron_ds[i] for i in pretectum_indices]))
mb_ds = np.nanmean(np.abs([neuron_ds[i] for i in midbrain_indices]))
hb_ds = np.nanmean(np.abs([neuron_ds[i] for i in hindbrain_indices]))

# Time constants
pt_time_constants = np.nanmean([neuron_tc[i] for i in pretectum_indices])
mb_time_constants = np.nanmean([neuron_tc[i] for i in midbrain_indices])
hb_time_constants = np.nanmean([neuron_tc[i] for i in hindbrain_indices])

# Reliability
neuron_rel = np.clip(neuron_rel, 0, 3)
pt_rel = np.nanmean([neuron_rel[i] for i in pretectum_indices])
mb_rel = np.nanmean([neuron_rel[i] for i in midbrain_indices])
hb_rel = np.nanmean([neuron_rel[i] for i in hindbrain_indices])

# Calculate standard deviations for error bars
from scipy.stats import sem
sem_ds = [sem(np.abs([neuron_ds[i] for i in pretectum_indices])),
          sem(np.abs([neuron_ds[i] for i in midbrain_indices])),
          sem(np.abs([neuron_ds[i] for i in hindbrain_indices]))]

sem_tc = [sem([neuron_tc[i] for i in pretectum_indices]),
          sem([neuron_tc[i] for i in midbrain_indices]),
          sem([neuron_tc[i] for i in hindbrain_indices])]

sem_rel = [sem([neuron_rel[i] for i in pretectum_indices]),
           sem([neuron_rel[i] for i in midbrain_indices]),
          sem([neuron_rel[i] for i in hindbrain_indices])]

# Values to plot in the specified order: Pretectum, Midbrain, Hindbrain
regions = ['Pretectum', 'Midbrain', 'Hindbrain']
avg_ds = [pt_ds, mb_ds, hb_ds]
avg_time_constants = [pt_time_constants, mb_time_constants, hb_time_constants]
avg_rel = [pt_rel, mb_rel, hb_rel]

# Define font settings
plt.rcParams.update({'font.family': 'Arial', 'font.size': 6})

# Define colors for bars
colors = ['#FF7F0E', '#1F77B4', '#2CA02C']  # Orange, blue, and green for visual contrast

# Create side-by-side subplots
fig, axs = plt.subplots(1, 3, figsize=(width_mm * 2, height_mm ))  # Convert mm to inches

# Plot for Direction Selectivity
axs[0].bar(regions, avg_ds, color=colors, yerr=sem_ds, capsize=3, width=0.4)
axs[0].set_ylabel("Average Direction Selectivity", fontsize=6)
axs[0].grid(False)

# Plot for Time Constant
axs[1].bar(regions, avg_time_constants, color=colors, yerr=sem_tc, capsize=3, width=0.4)
axs[1].set_ylabel("Average Time Constant", fontsize=6)
axs[1].grid(False)

# Plot for Reliability
axs[2].bar(regions, avg_rel, color=colors, yerr=sem_rel, capsize=3, width=0.4)
axs[2].set_ylabel("Average Reliability", fontsize=6)
axs[2].grid(False)

# Display the plot
plt.tight_layout()
file_name = 'bar_plots.pdf'
file_path = os.path.join(PATH_OUTPUT, file_name)
plt.savefig(file_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Figure saved to {file_path}")
plt.show()

# Time constant MW test
# Perform Mann-Whitney U test
stat, p_value = mannwhitneyu([neuron_tc[i] for i in pretectum_indices], [neuron_tc[i] for i in hindbrain_indices], alternative='two-sided')

# Output the test statistic and p-value
print("Mann-Whitney U test statistic:", stat)
print("p-value:", p_value)
