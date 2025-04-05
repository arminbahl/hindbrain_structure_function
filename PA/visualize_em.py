"""
Visualization of EM and PA Neuronal Data in Zebrafish Hindbrain
==============================================================

Author: Florian Kämpf
Date: April 6, 2025

Overview:
    This script visualizes neuronal data from electron microscopy (EM) and photoactivation (PA) experiments
    in the zebrafish hindbrain. It includes:
    - Loading and processing neuronal meshes from EM and PA datasets.
    - Assigning colors to neurons based on their functional types.
    - Generating interactive 3D visualizations of neurons and their connectivity.

Key Features:
    1. **Data Integration**:
        - Loads neuronal meshes from EM and PA datasets.
        - Combines data from seed cells and postsynaptic partners.
    2. **Categorization and Coloring**:
        - Assigns neurons to functional categories (e.g., integrator, motor command).
        - Maps functional categories to specific colors for visualization.
    3. **3D Visualization**:
        - Creates interactive 3D plots of neurons using Navis and Plotly.
        - Displays neurons with color-coded functional types.

Dependencies:
    - Python 3.7+
    - Required Libraries:
        * navis, numpy, matplotlib, pandas, plotly
        * Custom modules: `load_pa_table`, `load_mesh`

Data Requirements:
    - A structured folder containing EM and PA neuronal data (`path_to_data`).
    - Seed cell and postsynaptic partner data for EM neurons.

Outputs:
    - Interactive 3D plots of neurons (HTML files).

Usage Instructions:
    1. Update the paths for `path_to_data`, `path_seed_cells`, and `path_postsynaptic` to match your dataset.
    2. Run the script to:
        - Load and process neuronal meshes.
        - Assign functional categories and colors to neurons.
        - Generate 3D visualizations of neurons and their connectivity.
    3. View the results in the generated HTML file (`test.html`).

Notes:
    - Ensure that input data is properly formatted and follows the expected structure.
    - Remove duplicate keys in the `color_cell_type_dict` dictionary to avoid conflicts.
    - Adjust visualization parameters (e.g., colors, aspect ratio) as needed for specific analyses.
"""

import os
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import navis
import pandas as pd
import plotly

# Import custom modules for loading data and tools
from hindbrain_structure_function.visualization.FK_tools.load_pa_table import load_pa_table
from hindbrain_structure_function.visualization.FK_tools.load_mesh import load_mesh

# Set the current timestamp for logging or file naming
name_time = datetime.now()

# Define the path to the data directory
path_to_data = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data')

# Load the photoactivation (PA) table
pa_table = load_pa_table(path_to_data.joinpath("paGFP").joinpath("photoactivation_cells_table.csv"))
all_pa_cells = pa_table

# Initialize mesh-related columns in the PA table
for mesh_type in ['soma_mesh', 'dendrite_mesh', 'axon_mesh', 'neurites_mesh']:
    all_pa_cells[mesh_type] = np.nan
    all_pa_cells[mesh_type] = all_pa_cells[mesh_type].astype(object)

# Load meshes for each cell in the PA table
for i, cell in all_pa_cells.iterrows():
    all_pa_cells.loc[i, :] = load_mesh(cell, path_to_data, use_smooth_pa=True)

# Define paths for seed cells and postsynaptic cells
path_seed_cells = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\em_zfish1\data_seed_cells\output_data')
path_postsynaptic = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\em_zfish1\data_cell_89189_postsynaptic_partners\output_data')

# Initialize lists to store EM cells and their types
em_cells = []
seed_or_post = []

# Load seed cells
for cell in os.listdir(path_seed_cells):
    path_to_cell = path_seed_cells.joinpath(cell).joinpath('mapped').joinpath(f"em_fish1_{cell.split('_')[-1]}_mapped.obj")
    if path_to_cell.exists():
        em_cells.append(navis.read_mesh(path_to_cell, units="um"))
        seed_or_post.append('seed')
    else:
        print(f"{cell} not mapped")

# Load postsynaptic cells
for cell in os.listdir(path_postsynaptic):
    path_to_cell = path_postsynaptic.joinpath(cell).joinpath('mapped').joinpath(f"em_fish1_{cell.split('_')[-1]}_mapped.obj")
    if path_to_cell.exists():
        em_cells.append(navis.read_mesh(path_to_cell, units="um"))
        seed_or_post.append('post')
    else:
        print(f"{cell} not mapped")

# Define a dictionary for cell type colors
color_cell_type_dict = {
    "integrator": "red",
    "dynamic threshold": "cyan",
    "motor command": "purple",
    "motor_command": "purple",  # Duplicate key, consider removing
    "dynamic_threshold": "cyan",  # Duplicate key, consider removing
}

# Prepare cells and colors for visualization
visualized_cells = []
color_list = []

for i, cell in all_pa_cells.iterrows():
    visualized_cells.append(cell['neurites_mesh'])
    # Map cell type labels to colors
    temp = [x in color_cell_type_dict.keys() for x in cell['cell_type_labels']]
    if any(temp):
        color_list.append(color_cell_type_dict[(np.array(cell['cell_type_labels'])[temp])[0]])
    else:
        color_list.append('gray')  # Default color if no match is found

# Add EM cells to the visualization
for cell in em_cells:
    visualized_cells.append(cell)
    color_list.append('k')  # Black color for EM cells

# Create a 3D plot using Navis and Plotly
fig = navis.plot3d(visualized_cells, backend='plotly', color=color_list, width=1920, height=1080)
fig.update_layout(
    scene={
        'xaxis': {'autorange': True},
        'yaxis': {'autorange': True},
        'zaxis': {'autorange': True},
        'aspectmode': "data",
        'aspectratio': {"x": 1, "y": 1, "z": 1}
    }
)

# Save the plot as an HTML file and open it in the browser
plotly.offline.plot(fig, filename="test.html", auto_open=True, auto_play=False)