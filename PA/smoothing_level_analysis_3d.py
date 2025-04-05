import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import navis
import matplotlib
import plotly
from tqdm import tqdm
from copy import deepcopy
from hindbrain_structure_function.visualization.FK_tools.load_pa_table import load_pa_table
from hindbrain_structure_function.visualization.FK_tools.load_clem_table import load_clem_table
from hindbrain_structure_function.visualization.FK_tools.load_mesh import load_mesh

"""
3D Smoothing Level Analysis of Neuronal Skeletons
=================================================

Author: Florian Kämpf
Date: April 6, 2025

Overview:
    This script performs a 3D smoothing level analysis of neuronal skeletons from zebrafish hindbrain data.
    It includes:
    - Loading and processing neuronal data from photoactivation (PA) and correlative light and electron microscopy (CLEM) modalities.
    - Resampling and smoothing neuronal skeletons at various levels.
    - Comparing neuronal similarity using NBLAST.
    - Visualizing neurons in 3D with interactive plots.

Key Features:
    1. **Data Loading**:
        - Loads neuronal data from PA and CLEM datasets.
        - Combines data from multiple imaging modalities.
    2. **Skeleton Processing**:
        - Resamples neuronal skeletons to a uniform resolution.
        - Applies smoothing at different levels to analyze structural changes.
    3. **Similarity Analysis**:
        - Uses NBLAST to compute similarity between neurons.
        - Evaluates similarity across different smoothing levels.
    4. **3D Visualization**:
        - Generates interactive 3D plots of neurons.
        - Displays smoothed and original skeletons for comparison.

Dependencies:
    - Python 3.7+
    - Required Libraries:
        * navis, numpy, pandas, matplotlib, plotly, tqdm
        * Custom modules: `load_pa_table`, `load_clem_table`, `load_mesh`

Data Requirements:
    - A structured folder containing PA and CLEM neuronal data (`path_to_data`).
    - Neuronal skeletons in SWC format.

Outputs:
    - NBLAST similarity matrices.
    - Interactive 3D plots of neurons (HTML files).

Usage Instructions:
    1. Update the `path_to_data` variable to point to your dataset.
    2. Run the script to:
        - Load and process neuronal data.
        - Resample and smooth skeletons.
        - Perform similarity analysis using NBLAST.
        - Generate 3D visualizations of neurons.
    3. View the results in the generated HTML file (`test.html`).

Notes:
    - Ensure that input data is properly formatted and follows the expected structure.
    - Adjust smoothing levels and NBLAST parameters as needed for specific analyses.
    - Use the `make_figures_FK_output` directory to store output files.
"""

# Use TkAgg backend for matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    # Settings
    modalities = ['clem', 'pa']  # Imaging modalities to process
    name_time = datetime.now()  # Current timestamp for logging or file naming

    # Path settings
    path_to_data = Path(rf'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data')

    # Load data tables based on selected modalities
    if 'pa' in modalities:
        pa_table = load_pa_table(path_to_data.joinpath("paGFP").joinpath("photoactivation_cells_table.csv"))
    if 'clem' in modalities:
        clem_table = load_clem_table(path_to_data.joinpath('clem_zfish1').joinpath('all_cells'))

    # Combine tables if multiple modalities are selected
    if len(modalities) > 1:
        all_cells = pd.concat([eval(x + '_table') for x in modalities])
    elif len(modalities) == 1:
        all_cells = eval(modalities[0] + "_table")
    all_cells = all_cells.reset_index(drop=True)

    # Initialize SWC column for storing neuron skeletons
    all_cells['swc'] = np.nan
    all_cells['swc'] = all_cells['swc'].astype(object)

    # Set imaging modality to 'clem' for specific tracer
    all_cells.loc[all_cells['tracer_names'] == 'Jonathan Boulanger-Weill', 'imaging_modality'] = 'clem'

    # Load meshes for each cell that matches the selected modalities
    for i, cell in all_cells.iterrows():
        all_cells.loc[i, :] = load_mesh(cell, path_to_data, swc=True)

    # Subset to cells with valid SWC skeletons
    all_cells = all_cells[all_cells['swc'].apply(lambda x: isinstance(x, navis.core.skeleton.TreeNeuron))]

    # Resample neurons to a uniform resolution
    all_cells['swc'] = all_cells['swc'].apply(lambda x: navis.resample_skeleton(x, 0.1))

    # Create a dictionary to store smoothed versions of neurons
    smoothed_dict = {0: all_cells}
    for smoothing_level in np.arange(5, 100, 5):
        smoothed_dict[smoothing_level] = deepcopy(all_cells)
        smoothed_dict[smoothing_level]['swc'] = smoothed_dict[smoothing_level].apply(
            lambda x: navis.smooth_skeleton(x['swc'], window=smoothing_level)
            if x['imaging_modality'] == 'photoactivation' else x['swc'], axis=1
        )

    # Perform NBLAST to find the most similar neurons
    my_neuron_list = navis.NeuronList(all_cells.swc)
    dps = navis.make_dotprops(my_neuron_list, k=5, resample=False)
    nbl = navis.nblast(dps, dps, progress=False)
    nbl.index = all_cells.cell_name
    nbl.columns = all_cells.cell_name

    # Subset NBLAST results to exclude self-comparisons
    nbl_without_self = nbl.iloc[:(all_cells['imaging_modality'] == "clem").sum(),
                                 (all_cells['imaging_modality'] == "clem").sum():]

    # Find the most similar neurons
    indexer = nbl_without_self.max().sort_values(ascending=False).index
    clem_cell = all_cells[all_cells['cell_name'] == nbl_without_self.idxmax()[indexer][0]]
    pa_cell = all_cells[all_cells['cell_name'] == indexer[0]]

    # Evaluate similarity across smoothing levels
    list_values = []
    for smoothing_level in np.arange(5, 100, 5):
        my_neuron_list = navis.NeuronList([
            navis.smooth_skeleton(pa_cell['swc'].values[0], window=smoothing_level),
            clem_cell['swc'].values[0]
        ])
        dps = navis.make_dotprops(my_neuron_list, k=5, resample=False)
        nbl = navis.nblast(dps, dps, progress=False)
        nbl_array = np.array(nbl)
        list_values.append(nbl_array[1, 0])

    # Visualize the neurons in 3D
    fig = navis.plot3d(
        [navis.smooth_skeleton(pa_cell['swc'].values[0], window=smoothing_level), clem_cell['swc'].values[0]],
        backend='plotly', width=1920, height=1080
    )
    fig.update_layout(
        scene={
            'xaxis': {'autorange': True},
            'yaxis': {'autorange': True},
            'zaxis': {'autorange': True},
            'aspectmode': "data",
            'aspectratio': {"x": 1, "y": 1, "z": 1}
        }
    )

    # Save the 3D visualization as an HTML file
    output_dir = Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("html")
    os.makedirs(output_dir, exist_ok=True)
    plotly.offline.plot(fig, filename=str(output_dir.joinpath("test.html")), auto_open=True, auto_play=False)

