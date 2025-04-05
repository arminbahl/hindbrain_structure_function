"""
Visualization of Neuronal Cell Types in Zebrafish Hindbrain
==========================================================

Author: Florian Kämpf
Date: April 6, 2025

Overview:
    This script visualizes neuronal cell types in the zebrafish hindbrain based on their morphology,
    function, and neurotransmitter type. It includes:
    - Categorizing neurons by morphology, function, and neurotransmitter type.
    - Filtering and pruning neuronal structures for better visualization.
    - Generating interactive 3D plots of neurons and brain meshes.

Key Features:
    1. **Categorization**: Assigns neurons to categories such as ipsilateral/contralateral, inhibitory/excitatory,
       and functional types (e.g., integrator, motor command).
    2. **Filtering and Pruning**: Filters neurons by type and morphology and prunes neuronal structures
       to simplify visualization.
    3. **3D Visualization**:
        - Interactive 3D plots of neurons and brain meshes.
        - Color-coded neurons based on their functional types.

Dependencies:
    - Python 3.7+
    - Required Libraries:
        * navis, numpy, matplotlib, pandas, plotly
        * Custom modules: `get_base_path`, `load_cells_predictor_pipeline`, `load_brs`

Data Requirements:
    - A structured folder containing neuron data (`path_to_data`).
    - Predictor pipeline data for neuronal classification.

Outputs:
    - Interactive 3D plots of neurons and brain meshes (HTML files).

Usage Instructions:
    1. Ensure the base path is correctly set in `path_configuration.txt`.
    2. Run the script to:
        - Load neuronal data and classify neurons.
        - Filter and prune neurons by type and morphology.
        - Generate 3D visualizations of neurons and brain meshes.
    3. View the results in the generated HTML files.

Notes:
    - Ensure that input data is properly formatted and follows the expected structure.
    - Adjust pruning thresholds and neuron categories as needed for specific analyses.
"""

import navis
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from hindbrain_structure_function.visualization.FK_tools.get_base_path import get_base_path
from hindbrain_structure_function.visualization.FK_tools.load_pa_table import load_cells_predictor_pipeline
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import load_brs
import plotly

# Load base path and data
path_to_data = get_base_path()  # Ensure this path is set in path_configuration.txt
all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"])

# Define cell type categories
cell_type_categories = {
    'morphology': ['ipsilateral', 'contralateral'],
    'neurotransmitter': ['inhibitory', 'excitatory'],
    'function': ['integrator', 'dynamic_threshold', 'dynamic threshold', 'motor_command', 'motor command']
}

# Assign cell type labels to corresponding categories
for i, cell in all_cells_pa.iterrows():
    if isinstance(cell.cell_type_labels, list):
        for label in cell.cell_type_labels:
            if label in cell_type_categories['morphology']:
                all_cells_pa.loc[i, 'morphology'] = label
            elif label in cell_type_categories['function']:
                all_cells_pa.loc[i, 'function'] = label
            elif label in cell_type_categories['neurotransmitter']:
                all_cells_pa.loc[i, 'neurotransmitter'] = label
    # Append neurotransmitter type to the SWC name
    all_cells_pa.loc[i, 'swc'].name += f" NT: {all_cells_pa.loc[i, 'neurotransmitter']}"

# Load brain meshes
brain_meshes = load_brs(path_to_data, 'raphe')

# Helper function to prune and filter cells by type and morphology
def filter_and_prune_cells(function_type, morphology_type, prune_threshold=20):
    """
    Filters and prunes cells based on their function and morphology.

    Args:
        function_type (str): The functional type of the cells (e.g., 'integrator').
        morphology_type (str): The morphological type of the cells (e.g., 'contralateral').
        prune_threshold (int): The threshold for pruning twigs from the cells.

    Returns:
        list: A list of pruned cells.
    """
    cells = list(all_cells_pa.loc[
        (all_cells_pa['function'] == function_type) & 
        (all_cells_pa['morphology'] == morphology_type), 'swc'
    ])
    return [navis.prune_twigs(cell, prune_threshold, recursive=True) for cell in cells]

# Filter and prune cells by function and morphology
contralateral_integrators = filter_and_prune_cells('integrator', 'contralateral')
contralateral_dynamic_threshold = filter_and_prune_cells('dynamic_threshold', 'contralateral')
contralateral_motor_command = filter_and_prune_cells('motor_command', 'contralateral')

# Plot 3D visualization of cells and brain meshes
fig = navis.plot3d(contralateral_integrators + brain_meshes, backend='plotly',
                   width=1920, height=1080, hover_name=True, colors='red')

fig = navis.plot3d(contralateral_dynamic_threshold, backend='plotly', fig=fig,
                   width=1920, height=1080, hover_name=True, colors='blue')

fig = navis.plot3d(contralateral_motor_command, backend='plotly', fig=fig,
                   width=1920, height=1080, hover_name=True, colors='purple')

# Update layout for better visualization
fig.update_layout(
    scene={
        'xaxis': {'autorange': 'reversed'},  # Reverse x-axis
        'yaxis': {'autorange': True},
        'zaxis': {'autorange': True},
        'aspectmode': "data",
        'aspectratio': {"x": 1, "y": 1, "z": 1}
    }
)

# Save the plot to an HTML file
plotly.offline.plot(fig, filename="test.html", auto_open=True, auto_play=False)

# Example: Prune and visualize a specific test cell
test_cell = all_cells_pa.loc[all_cells_pa['cell_name'] == '20230418.1', 'swc'].iloc[0]
pruned_test_cell = [navis.prune_twigs(test_cell, 20, recursive=True)]

fig = navis.plot3d(pruned_test_cell + brain_meshes, backend='plotly',
                   width=1920, height=1080, hover_name=True, colors='red')

fig.update_layout(
    scene={
        'xaxis': {'autorange': 'reversed'},  # Reverse x-axis
        'yaxis': {'autorange': True},
        'zaxis': {'autorange': True},
        'aspectmode': "data",
        'aspectratio': {"x": 1, "y": 1, "z": 1}
    }
)

# Save the test cell plot to an HTML file
plotly.offline.plot(fig, filename="dt.html", auto_open=True, auto_play=False)

