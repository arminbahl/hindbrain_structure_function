"""
Neural Connectivity Analysis Script for Zebrafish Hindbrain Neurons
===================================================================

Author: Jonathan Boulanger-Weill
Date: November 26, 2024

Overview:
    This script conducts a comprehensive analysis of the neural connectivity of traced neurons
    within the zebrafish hindbrain. The analysis includes:
    - Classifying hemispheres for traced neurons.
    - Extracting and analyzing synaptic inputs and outputs.
    - Generating connectivity matrices categorized by functional neuron types.
    - Visualizing connectivity through plots and mesh displays.
    - Producing activity traces for functionally imaged neurons.

Key Features:
    1. **Hemispheric Classification**: Determine whether neurons are ipsilateral or contralateral based on synaptic data.
    2. **Connectivity Matrices**: Quantify the directional connections (inputs and outputs) between functional neuron groups.
    3. **Visualization**:
        - Neural network connectivity plots.
        - Activity traces of neurons under different stimuli conditions.
        - Brain region and connectome mesh overlays.
    4. **Dynamic Traces**: Analyze neuron activity over time for various functional types.

Dependencies:
    - Python 3.7+
    - Required Libraries:
        * pandas, numpy, matplotlib, h5py, openpyxl, scipy
        * Custom modules: `connectome_helpers_current`, `FK_tools`

Data Requirements:
    - A structured folder containing neuron and synaptic data (`ROOT_FOLDER`).
    - Hemisphere classification data (`OUTPUT_CSV`).
    - Excel file with traced neuron metadata (`PATH_ALL_CELLS`).

Outputs:
    - CSV files with updated hemispheric classifications.
    - Connectivity matrices (directional).
    - Neural network diagrams (PDFs).
    - Mesh visualizations for traced neurons.
    - Activity plots of functionally imaged neurons.

Usage Instructions:
    1. Update file paths under the "Paths" section of the script to match your dataset.
    2. Run the script in its entirety to:
        - Generate hemispheric classifications.
        - Fetch neuron IDs and connectivity data.
        - Compute and visualize directional connectivity matrices.
    3. View the results in the specified output folders.

Notes:
    - Ensure that input files are properly formatted and follow the expected structure.
    - Adjust seed cell IDs and parameters under the respective sections to customize the analysis.

"""
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import h5py
from scipy.signal import savgol_filter
from datetime import datetime
from getch import getch
import sys
import os

sys.path.append("/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/connectomes")

from connectome_helpers_current import (  # Import helper functions
    determine_hemisphere,
    fetch_filtered_ids, 
    get_inputs_outputs_by_hemisphere,
    compute_count_probabilities_from_results,
    draw_two_layer_neural_net,
    get_outputs_inputs_neurons, 
    plot_output_connectome, 
    plot_input_connectome, 
    COLOR_CELL_TYPE_DICT, 
    find_and_load_cell_meshes, 
    plot_neuron_activity,
    setup_plot, 
)

# Paths
ROOT_FOLDER = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/traced_neurons/all_cells_111224/'
PATH_ALL_CELLS = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224.csv'
PATH_ALL_CELLS_HEMISPHERE = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224_with_hemisphere.csv'
OUTPUT_PATH_NETWORKS = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/connectomes/nn_diagrams'
OUTPUT_PATH_NUCLEI_IDs = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/connectomes/nuclei_ids_txt'
OUTPUT_PATH_ACTIVITY = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/connectomes/activity'

# Load data
all_cells_xls = pd.read_csv(PATH_ALL_CELLS)
all_cells_xls_hemisphere = pd.read_csv(PATH_ALL_CELLS_HEMISPHERE)
# Remove duplicates
all_cells_xls_hemisphere = all_cells_xls_hemisphere. drop_duplicates(subset='axon_id')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 1: Classify hemispheres

# Initialize progress tracker
progress = {'processed_count': 0, 'total_rows': len(all_cells_xls)}
all_cells_xls_hemisphere['hemisphere'] = all_cells_xls.apply(determine_hemisphere, axis=1, root_folder=ROOT_FOLDER, progress=progress)
all_cells_xls_hemisphere.to_csv(PATH_ALL_CELLS_HEMISPHERE, index=False)
print(f"Updated DataFrame saved to: {PATH_ALL_CELLS_HEMISPHERE}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 2: Fetch neuron IDs for analysis

# Standardize namings 
all_cells_xls_hemisphere['functional classifier'] = all_cells_xls_hemisphere['functional classifier'].replace({
    'dynamic threshold': 'dynamic_threshold',
    'motor command': 'motor_command'
})

# All IDs 
dt_ids_all_nuc, dt_ids_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere, 9, 'dynamic_threshold'); 
ic_ids_all_nuc,  ic_ids_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere, 9, 'integrator', 11, 'contralateral')
ii_ids_all_nuc, ii_ids_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere, 9, 'integrator', 11, 'ipsilateral')
mc_ids_all_nuc, mc_ids_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere, 9, 'motor_command')

# Pooled IDs
all_ids_nuc = np.concatenate([dt_ids_all_nuc, ic_ids_all_nuc, ii_ids_all_nuc, mc_ids_all_nuc])
all_ids_fun = np.concatenate([dt_ids_all_fun, ic_ids_all_fun, ii_ids_all_fun, mc_ids_all_fun])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 3: Export connectomes 

connectome = get_inputs_outputs_by_hemisphere(
    root_folder=ROOT_FOLDER,
    seed_cell_ids=['576460752678870154'],
    hemisphere_df=all_cells_xls_hemisphere
)

# Extract Outputs
same_side_outputs_cells = connectome["outputs"]["cells"]["same_side"]
different_side_outputs_cells = connectome["outputs"]["cells"]["different_side"]
same_side_outputs_synapses = connectome["outputs"]["synapses"]["same_side"]
different_side_outputs_synapses = connectome["outputs"]["synapses"]["different_side"]
outputs_percentages = connectome["outputs"]["percentages"]

# Extract Inputs
same_side_inputs_cells = connectome["inputs"]["cells"]["same_side"]
different_side_inputs_cells = connectome["inputs"]["cells"]["different_side"]
same_side_inputs_synapses = connectome["inputs"]["synapses"]["same_side"]
different_side_inputs_synapses = connectome["inputs"]["synapses"]["different_side"]
inputs_percentages = connectome["inputs"]["percentages"]

# Extract Counters
output_seed_counter = connectome["counters"]["output_seed_counter"]
input_seed_counter = connectome["counters"]["input_seed_counter"]

# Statistics for the text: 
pooled_connectomes = pd.concat([same_side_outputs_synapses, different_side_outputs_synapses, same_side_inputs_synapses, different_side_inputs_synapses])
pooled_connectomes = pooled_connectomes. drop_duplicates(subset='axon_id')
partners_all_nuc, partners_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere, 1, 'not functionally imaged'); 
num_axons = len(pooled_connectomes.loc[pooled_connectomes['type'] == 'axon'])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 4: Compute probabilities 
probabilities_results = compute_count_probabilities_from_results(connectome, functional_only=False)

# Access probabilities for same-side outputs (cells)
same_side_outputs_cells_cat = probabilities_results["outputs"]["same_side"]["cells"]
same_side_outputs_synapses_cat = probabilities_results["outputs"]["same_side"]["synapses"]
same_side_inputs_cells_cat = probabilities_results["inputs"]["same_side"]["cells"]
same_side_inputs_synapses_cat = probabilities_results["inputs"]["same_side"]["synapses"]

different_side_outputs_cells_cat = probabilities_results["outputs"]["different_side"]["cells"]
different_side_outputs_synapses_cat = probabilities_results["outputs"]["different_side"]["synapses"]
different_side_inputs_cells_cat = probabilities_results["inputs"]["different_side"]["cells"]
different_side_inputs_synapses_cat = probabilities_results["inputs"]["different_side"]["synapses"]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 5: Draw neural network

fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Adjusted figsize for compactness

# Titles for each subplot
titles = [
    "Same Side Inputs Synapses",
    "Different Side Inputs Synapses",
    "Same Side Outputs Synapses",
]

# DataFrames corresponding to each subplot
dataframes = [
    same_side_inputs_synapses_cat,
    different_side_inputs_synapses_cat,
    same_side_outputs_synapses_cat,
]

# Connection type for each DataFrame
connection_types = [
    "inputs",  # Same Side Inputs Synapses
    "inputs",  # Different Side Inputs Synapses
    "outputs",  # Same Side Outputs Synapses
]

# Whether to show the midline based on the subplot's context
show_midlines = [
    False,  # Same Side Inputs Synapses
    True,   # Different Side Inputs Synapses
    False,  # Same Side Outputs Synapses
]

# Loop through subplots and draw the network for each
for ax, title, data_df, connection_type, show_midline in zip(axes.flatten(), titles, dataframes, connection_types, show_midlines):
    draw_two_layer_neural_net(
        ax=ax,
        left=0.1, right=0.6, bottom=0.5, top=1.1,
        data_df=data_df,
        node_radius=0.02,  # Larger node size
        input_circle_color='integrator_ipsilateral',  # Input node color #'integrator_contralateral'
        input_cell_type='mixed',  # Excitatory connections with arrows
        show_midline=show_midline,  # Show the midline only for "different side" plots
        proportional_lines=True,  # Proportional connection line thickness
        a=6, b=2,  # Scale factors for line and arrow sizes
        connection_type=connection_type  # Set the connection type (inputs or outputs)
    )
    ax.set_title(title, fontsize=14)

# Adjust spacing between subplots for compactness
plt.tight_layout()

# Show and save the figure
plot_str = "ii_all"
output_pdf_path = os.path.join(OUTPUT_PATH_NETWORKS, f"neural_network_visualization_111224_{plot_str}.pdf")
plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
plt.show()
print(f"Visualization saved as: {output_pdf_path}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 6: Generate connectomes meshes displays 

# Load Florian Kampf's helper function 
os.chdir('/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/structure_function')
from FK_tools.load_brs import *
from FK_tools.get_base_path import *
path_to_data = get_base_path() 

# Example DT connectome for Main Fig. 2 and SV Extended Fig. 5
seed_cell_ids = ['576460752734987800'] 
plot_stg='576460752734987800'
colors_sc = [(100/255, 197/255, 235/255, 0.7)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=None)

# Several DTs connectome for Main Fig. 2
seed_cell_ids = ['576460752681311362', '576460752718169177', '576460752734987800'] 
plot_stg='dts'
colors_sc = [(100/255, 197/255, 235/255, 0.7)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=plot_stg)

# Example CI connectome for Main Fig. 2
seed_cell_ids = ['576460752680588674'] 
plot_stg='576460752680588674'
colors_sc = [(232/255, 77/255, 138/255, 0.7)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=plot_stg)

# Several CIs connectome for Main Fig. 2
seed_cell_ids = ['576460752680588674', '576460752680445826'] 
plot_stg='ics'
colors_sc = [(232/255, 77/255, 138/255, 0.7)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=plot_stg)

# Example MC connectome for Main Fig. 2 
seed_cell_ids = ['576460752684182585']  
plot_stg='576460752684182585'
colors_sc = [(127/255, 88/255, 175/255, 0.7)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=plot_stg)
 
# Example II connectome for Main Fig. 2 
seed_cell_ids = ['576460752631366630'] #4548
plot_stg='576460752631366630'
colors_sc = [(254/255, 179/255, 38/255)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=plot_stg)

# Example #2 II connectome for Main Fig. 2 
seed_cell_ids = ['576460752741561977'] #2939
plot_stg='576460752741561977'
colors_sc = [(254/255, 179/255, 38/255)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=plot_stg)
# Remove the 'noisy neurons'
output_neurons = output_neurons[output_neurons['functional classifier'] != 'noisy, little modulation']
input_neurons = input_neurons[input_neurons['functional classifier'] != 'noisy, little modulation']

# Example #3 II connectome for Main Fig. 2 
seed_cell_ids = ['576460752671949300'] #3862
plot_stg='576460752671949300'
colors_sc = [(254/255, 179/255, 38/255)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=plot_stg)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_stg='ii_3'

# Drop duplicates
output_neurons = output_neurons.drop_duplicates(subset=['axon_id'])
input_neurons = input_neurons.drop_duplicates(subset=['axon_id'])

# Load meshes 
axon_sc, dendrite_sc, soma_sc = find_and_load_cell_meshes(ROOT_FOLDER, seed_cell_ids)
mesh_sc = axon_sc + dendrite_sc + soma_sc

# Get background brain region meshes 
which_brs='raphe'
brain_meshes = load_brs(path_to_data, which_brs=which_brs)
color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Output connectome 
meshes_rec=[]; meshes_notrec=[]; 
colors_rec=[]; colors_notrec=[]; 

for cell_idx in range(len(output_neurons)):
    # Load meshes
    cell_id = output_neurons.iloc[cell_idx, 5]

    # Not functionally recorded 
    if output_neurons.iloc[cell_idx, 1] == 'not functionally imaged':  

        # Myelinated neuron
        if output_neurons.iloc[cell_idx, 9]=="myelinated": 
                
            axon_mesh, dendrite_mesh, soma_mesh=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])

            colors_soma = [COLOR_CELL_TYPE_DICT.get("myelinated")]
            colors_dendrites = [COLOR_CELL_TYPE_DICT.get("myelinated")]
            colors_axon = [COLOR_CELL_TYPE_DICT.get("myelinated")]

            # Merge all recorded meshes and colors
            meshes_rec = meshes_rec + soma_mesh + dendrite_mesh + axon_mesh
            colors_rec = colors_rec + colors_soma + colors_dendrites + colors_axon

        else: 
            axon_mesh_notrec, dendrite_mesh_notrec, soma_mesh_notrec=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])
            colors_soma_notrec = [(0, 0, 0, 0.2)]
            colors_dendrites_notrec = [(0, 0, 0, 0.2)]
            colors_axon_notrec = [(0, 0, 0, 0.2)]

            # Merge all recorded meshes and colors
            meshes_notrec = meshes_notrec + soma_mesh_notrec + dendrite_mesh_notrec + axon_mesh_notrec
            colors_notrec = colors_notrec + colors_soma_notrec + colors_dendrites_notrec + colors_dendrites_notrec
       
    # Recorded
    elif output_neurons.iloc[cell_idx, 1] != 'not functionally imaged':
        axon_mesh, dendrite_mesh, soma_mesh=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])

        if output_neurons.iloc[cell_idx, 9]=='integrator' and output_neurons.iloc[cell_idx, 11]=='ipsilateral': 
            colors_soma = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]
            colors_dendrites = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]
            colors_axon = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]

        elif output_neurons.iloc[cell_idx, 9]=='integrator' and output_neurons.iloc[cell_idx, 11]=='contralateral': 
            colors_soma = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]
            colors_dendrites = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]
            colors_axon = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]      

        elif output_neurons.iloc[cell_idx, 9]=='dynamic_threshold': 
            colors_soma = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]
            colors_dendrites = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]
            colors_axon = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]

        elif output_neurons.iloc[cell_idx, 9]=='motor_command': 
            colors_soma = [COLOR_CELL_TYPE_DICT.get("motor_command")]
            colors_dendrites = [COLOR_CELL_TYPE_DICT.get("motor_command")]
            colors_axon = [COLOR_CELL_TYPE_DICT.get("motor_command")]
        else: 
            colors_soma = [(0, 0, 0, 0.2)]
            colors_dendrites = [(0, 0, 0, 0.2)]
            colors_axon = [(0, 0, 0, 0.2)]

        # Merge all recorded meshes and colors
        meshes_rec = meshes_rec + soma_mesh + dendrite_mesh + axon_mesh
        colors_rec = colors_rec + colors_soma + colors_dendrites + colors_axon

# Make output connectome plot 
projection='y'
plot_output_connectome(projection, brain_meshes, color_meshes, meshes_notrec, colors_notrec, meshes_rec+mesh_sc, colors_rec+colors_sc, which_brs, path_to_data, plot_stg)

projection='z'
plot_output_connectome(projection, brain_meshes, color_meshes, meshes_notrec, colors_notrec, meshes_rec+mesh_sc, colors_rec+colors_sc, which_brs, path_to_data, plot_stg)
 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input connectome 
meshes_rec=[]; meshes_notrec=[]; meshes_axons=[]; 
colors_rec=[]; colors_notrec=[]; colors_axons=[]; 

for cell_idx in range(len(input_neurons)):

    # Case axon caudal 
    if input_neurons.iloc[cell_idx, 0] == 'axon' and 'caudal' in input_neurons.iloc[cell_idx, 3]: 

        axon_id=input_neurons.iloc[cell_idx, 7]
        axon_mesh = find_and_load_cell_meshes(ROOT_FOLDER, [str(axon_id)])

        meshes_axons = meshes_axons + list(axon_mesh)
        colors_axons = colors_axons + [COLOR_CELL_TYPE_DICT.get("axon")]

    # Case axon rostral
    elif input_neurons.iloc[cell_idx, 0] == 'axon' and 'rostral' in input_neurons.iloc[cell_idx, 3]: 

        axon_id=input_neurons.iloc[cell_idx, 7]
        axon_mesh = find_and_load_cell_meshes(ROOT_FOLDER, [str(axon_id)])

        meshes_axons = meshes_axons + list(axon_mesh)
        colors_axons = colors_axons + [COLOR_CELL_TYPE_DICT.get("axon")]

    # Case neuron
    elif input_neurons.iloc[cell_idx, 0] == 'cell': 

         # Load meshes
        cell_id=input_neurons.iloc[cell_idx, 5]

        # Not functionally recorded 
        if input_neurons.iloc[cell_idx, 1] == 'not functionally imaged':  

            # Myelinated neuron
            if input_neurons.iloc[cell_idx, 9]=="myelinated": 
                    
                axon_mesh, dendrite_mesh, soma_mesh=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])

                colors_soma = [COLOR_CELL_TYPE_DICT.get("myelinated")]
                colors_dendrites = [COLOR_CELL_TYPE_DICT.get("myelinated")]
                colors_axon = [COLOR_CELL_TYPE_DICT.get("myelinated")]

                # Merge all recorded meshes and colors
                meshes_notrec = meshes_notrec + soma_mesh + dendrite_mesh + axon_mesh
                colors_notrec = colors_notrec + colors_soma + colors_dendrites + colors_axon

            else: 
                axon_mesh_notrec, dendrite_mesh_notrec, soma_mesh_notrec=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])
                colors_soma_notrec = [(0, 0, 0, 0.2)]
                colors_dendrites_notrec = [(0, 0, 0, 0.2)]
                colors_axon_notrec = [(0, 0, 0, 0.2)]

                # Merge all recorded meshes and colors
                meshes_notrec = meshes_notrec + soma_mesh_notrec + dendrite_mesh_notrec + axon_mesh_notrec
                colors_notrec = colors_notrec + colors_soma_notrec + colors_dendrites_notrec + colors_dendrites_notrec
        
        # Recorded
        elif input_neurons.iloc[cell_idx, 1] != 'not functionally imaged':
            axon_mesh, dendrite_mesh, soma_mesh=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])

            if input_neurons.iloc[cell_idx, 9]=='integrator' and input_neurons.iloc[cell_idx, 11]=='ipsilateral': 
                colors_soma = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]
                colors_dendrites = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]
                colors_axon = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]

            elif input_neurons.iloc[cell_idx, 9]=='integrator' and input_neurons.iloc[cell_idx, 11]=='contralateral': 
                colors_soma = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]
                colors_dendrites = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]
                colors_axon = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]      

            elif input_neurons.iloc[cell_idx, 9]=='dynamic_threshold': 
                colors_soma = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]
                colors_dendrites = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]
                colors_axon = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]

            elif input_neurons.iloc[cell_idx, 9]=='motor_command': 
                colors_soma = [COLOR_CELL_TYPE_DICT.get("motor_command")]
                colors_dendrites = [COLOR_CELL_TYPE_DICT.get("motor_command")]
                colors_axon = [COLOR_CELL_TYPE_DICT.get("motor_command")]
            else: 
                colors_soma = [(0, 0, 0, 0.2)]
                colors_dendrites = [(0, 0, 0, 0.2)]
                colors_axon = [(0, 0, 0, 0.2)]

            # Merge all recorded meshes and colors
            meshes_rec = meshes_rec + soma_mesh + dendrite_mesh + axon_mesh
            colors_rec = colors_rec + colors_soma + colors_dendrites + colors_axon
        
# Make input connectome plot 
projection='y'
plot_input_connectome(projection, brain_meshes, color_meshes, meshes_axons, colors_axons, 
                      meshes_notrec, colors_notrec, meshes_rec+mesh_sc, colors_rec+colors_sc, which_brs, path_to_data, plot_stg)
projection='z'
plot_input_connectome(projection, brain_meshes, color_meshes, meshes_axons, colors_axons, 
                      meshes_notrec, colors_notrec, meshes_rec+mesh_sc, colors_rec+colors_sc, which_brs, path_to_data, plot_stg)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 7: Activity traces  

# Example II #1 connectome for Main Fig. 2
seed_cell_ids = ['576460752631366630'] 
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=None)

# Example II #3 connectome for Main Fig. 2
seed_cell_ids = ['576460752671949300'] 
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=None)

# Example CI connectome for Main Fig. 2
seed_cell_ids = ['576460752680588674', '576460752680445826'] 
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=None)

# Example DT connectome for Main Fig. 2 and SV Extended Fig. 5
seed_cell_ids = ['576460752734987800']
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=None)

# Example MC connectome for Main Fig. 2 
seed_cell_ids = ['576460752684182585']  
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, PATH_ALL_CELLS, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=None)
 
# Function to set up and format the plot
all_cells = pd.read_csv(PATH_ALL_CELLS)
all_cells['functional classifier'] = all_cells['functional classifier'].replace({
    'dynamic threshold': 'dynamic_threshold',
    'motor command': 'motor_command'
})

# Load necessary data
all_cells_df = pd.read_csv(PATH_ALL_CELLS)

# Initialize an empty list to store the extracted seed cell function IDs
seed_cell_funct_ids = []

# Loop through each element in seed_cell_ids
for seed_cell_id in seed_cell_ids:
    # Find the seed cell function ID for the current seed cell ID
    try:
        seed_cell_funct_id = int(all_cells_df[all_cells_df.apply(
            lambda row: row.astype(str).str.contains(str(seed_cell_id), case=False, na=False).any(), axis=1
        )].iloc[:, 1].iloc[0])
        # Append the function ID to the list
        seed_cell_funct_ids.append(seed_cell_funct_id)
    except IndexError:
        print(f"Seed cell ID {seed_cell_id} not found.")

with h5py.File("/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/all_cells.h5", "r") as hdf_file:
    
    time_axis = None
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))  # Larger figure size to accommodate square subplots

    # Loop through all seed cell function IDs
    for seed_cell_funct_id in seed_cell_funct_ids:
        neuron_group = hdf_file[f"neuron_{seed_cell_funct_id}"]

        avg_activity_left = neuron_group["average_activity_left"][()]
        avg_activity_right = neuron_group["average_activity_right"][()]

        smooth_avg_activity_left = savgol_filter(avg_activity_left, 20, 3)
        smooth_avg_activity_right = savgol_filter(avg_activity_right, 20, 3)

        # Generate time axis based on the length of the activity data
        if time_axis is None:
            time_axis = np.arange(len(avg_activity_left)) * 0.5

        # Output connectome plots for the current seed cell
        plot_neuron_activity(axs[0, 0], output_neurons, time_axis, hdf_file, "left", COLOR_CELL_TYPE_DICT, [seed_cell_funct_id])
        plot_neuron_activity(axs[0, 1], output_neurons, time_axis, hdf_file, "right", COLOR_CELL_TYPE_DICT, [seed_cell_funct_id])

        # Input connectome plots for the current seed cell
        plot_neuron_activity(axs[1, 0], input_neurons, time_axis, hdf_file, "left", COLOR_CELL_TYPE_DICT, [seed_cell_funct_id])
        plot_neuron_activity(axs[1, 1], input_neurons, time_axis, hdf_file, "right", COLOR_CELL_TYPE_DICT, [seed_cell_funct_id])

        # Ensure the seed cell trace is plotted on top with its proper color
        setup_plot(axs[0, 0], f'Left Stimuli Output Activity for cell {seed_cell_funct_id}', time_axis, smooth_avg_activity_left, all_cells, COLOR_CELL_TYPE_DICT, [seed_cell_funct_id])
        setup_plot(axs[0, 1], f'Right Stimuli Output Activity for cell {seed_cell_funct_id}', time_axis, smooth_avg_activity_right, all_cells, COLOR_CELL_TYPE_DICT, [seed_cell_funct_id])
        setup_plot(axs[1, 0], f'Left Stimuli Input Activity for cell {seed_cell_funct_id}', time_axis, smooth_avg_activity_left, all_cells, COLOR_CELL_TYPE_DICT, [seed_cell_funct_id])
        setup_plot(axs[1, 1], f'Right Stimuli Input Activity for cell {seed_cell_funct_id}', time_axis, smooth_avg_activity_right, all_cells, COLOR_CELL_TYPE_DICT, [seed_cell_funct_id])

    plt.tight_layout(pad=3.0)  # Added padding for better spacing between plots
    plt.show()

    # Save the figure, including all seed cell function IDs in the filename
    name_time = datetime.now()
    filename = f"activity_in_out_connectome_sc_{'_'.join(map(str, seed_cell_funct_ids))}_{name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    file_path = os.path.join(OUTPUT_PATH_ACTIVITY, filename)
    fig.savefig(file_path, dpi=1200)
    print(f"Figure saved successfully at: {file_path}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 8: Import LDA results 

# Load LDA predictions 
lda_lxs = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/lda_nblast_predictions/clem_cell_prediction_optimize_all_predict_2025-02-27_16-52-09.xlsx'
lda_predictions = pd.read_excel(lda_lxs)

# Load original data to complete 
all_cells_xls_hemisphere = pd.read_csv(PATH_ALL_CELLS_HEMISPHERE)
# Standardize namings 
all_cells_xls_hemisphere['functional classifier'] = all_cells_xls_hemisphere['functional classifier'].replace({
    'dynamic threshold': 'dynamic_threshold',
    'motor command': 'motor_command'
})
# Add an LDA status column (base is native, when predicted becomes 'predicted') 
all_cells_xls_hemisphere['lda'] = 'native'

# Standardize the column names for matching
all_cells_xls_hemisphere['nucleus_id'] = all_cells_xls_hemisphere['nucleus_id'].astype(str).str.strip()
lda_predictions['cell_name'] = lda_predictions['cell_name'].str.replace('cell_', '', regex=False).astype(str).str.strip()

# Count how many unique axon IDs correspond to the selected cells
count_cells_to_predict = all_cells_xls_hemisphere.loc[
    (all_cells_xls_hemisphere['type'] == 'cell') & 
    (all_cells_xls_hemisphere['functional_id'] == 'not functionally imaged') & 
    (~all_cells_xls_hemisphere['functional classifier'].isin([
        'myelinated', 
        'noisy, little modulation', 
        'non-direction-selective, on response', 
        'off-response'
    ]))
]['axon_id'].nunique()

print(f"Number of unique 'axon_id' values with 'type' = 'cell', 'functional_id' = 'not functionally imaged', and excluding 'myelinated' in 'functional classifier': {count_cells_to_predict}")

# Initialize a counter for assigning unique integers to functional_id
functional_id_counter = 1

# Iterate through all_cells_xls_hemisphere and update 'functional_id'
for idx, row in all_cells_xls_hemisphere.iterrows():
    if row['functional_id'] == 'not functionally imaged': 

        # Match nucleus_id from all_cells_xls_hemisphere with cell_name in lda_predictions
        matching_row = lda_predictions[lda_predictions['cell_name'] == row['nucleus_id']]
        
        if not matching_row.empty:
            # Fetch the 'prediction' value
            predicted_functional_id = matching_row.iloc[0]['prediction']

            # Fetch the test results  
            pooled_tests = matching_row.iloc[0]['passed_tests'] 

            # Check if all tests pass
            if str(pooled_tests).strip().upper() == 'TRUE':  

                # Update 'functional classifier' based on the prediction
                if predicted_functional_id in ['integrator_ipsilateral', 'integrator_contralateral']:
                    all_cells_xls_hemisphere.at[idx, 'functional classifier'] = 'integrator'
                elif predicted_functional_id in ['dynamic_threshold', 'motor_command']:
                    all_cells_xls_hemisphere.at[idx, 'functional classifier'] = predicted_functional_id
                all_cells_xls_hemisphere.at[idx, 'lda'] = 'predicted'
            
                # Assign an integer ID to the 'functional_id' column
                all_cells_xls_hemisphere.at[idx, 'functional_id'] = functional_id_counter
                functional_id_counter += 1

                print(functional_id_counter)
                print(predicted_functional_id)
                print(f"Tests passed for nucleus_id {row['nucleus_id']}. Update performed.") 
 
# Display the updated DataFrame 
print(all_cells_xls_hemisphere.head())

# Save the updated DataFrame if needed
lda_csv_path = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224_with_hemisphere_lda_lof_022825.csv"
all_cells_xls_hemisphere.to_csv(lda_csv_path, index=False)
print(f"Updated DataFrame saved to: {lda_csv_path}")

# %% 
# Predicted IDs and Total Count
unique_nucleus_ids = all_cells_xls_hemisphere.loc[
    all_cells_xls_hemisphere['lda'] == 'predicted', 'nucleus_id'
].unique()

print(f"Unique predicted nucleus IDs: {unique_nucleus_ids}")
print(f"Total number of unique predicted nucleus IDs: {len(unique_nucleus_ids)}")

# Percentage of Prediction
percentage_predicted = (len(unique_nucleus_ids) / count_cells_to_predict) * 100
print(f"Percentage of predicted neurons: {percentage_predicted:.2f}%")

# Number of Unique 'cell_name' with 'TRUE' in 'passed_tests'
unique_cell_name_count = lda_predictions.loc[
    lda_predictions['passed_tests'].astype(str).str.strip().str.upper() == 'TRUE', 'cell_name'
].nunique()

print(f"Number of unique 'cell_name' values with 'passed_tests' = 'TRUE': {unique_cell_name_count}")

# Occurrences of Each Functional Classifier
# Filter rows where 'lda' is 'predicted' and drop duplicates based on unique identifiers
filtered_cells = all_cells_xls_hemisphere.loc[
    all_cells_xls_hemisphere['lda'] == 'predicted'
].drop_duplicates(subset=['nucleus_id'])

# Initialize a dictionary to store counts
functional_classifier_counts = {
    # Count for 'integrator' with 'contralateral' projection
    'integrator_contralateral': filtered_cells.loc[
        (filtered_cells['functional classifier'] == 'integrator') &
        (filtered_cells['projection classifier'] == 'contralateral')
    ].shape[0],

    # Count for 'integrator' with 'ipsilateral' projection
    'integrator_ipsilateral': filtered_cells.loc[
        (filtered_cells['functional classifier'] == 'integrator') &
        (filtered_cells['projection classifier'] == 'ipsilateral')
    ].shape[0]
}

# Count all other functional classifiers (excluding 'integrator')
other_classifiers = filtered_cells.loc[
    filtered_cells['functional classifier'] != 'integrator', 'functional classifier'
].value_counts()

# Add other classifiers to the dictionary
functional_classifier_counts.update(other_classifiers.to_dict())

# Print the results
print("\nOccurrences of each functional classifier:")
for classifier, count in functional_classifier_counts.items():
    print(f"{classifier}: {count}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 9: Regenerate enhanced connectomes 

# Load Florian Kampf's helper function 
os.chdir('/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/structure_function')
from FK_tools.load_brs import *
from FK_tools.get_base_path import *
path_to_data = get_base_path() 

lda_csv_path = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224_with_hemisphere_lda_lof_022825.csv'

# Example DT connectome for SV Extended Fig. 7
seed_cell_ids = ['576460752734987800'] 
ids_stg='576460752734987800_lda_022825'
plot_stg='dt_lda'
colors_sc = [(100/255, 197/255, 235/255, 0.7)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, lda_csv_path, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=ids_stg)

# Several DTs connectome for Extended Data Fig. 7
seed_cell_ids = ['576460752681311362', '576460752718169177', '576460752734987800'] 
ids_stg='dts_lda'
plot_stg='dts_lda_022825'
colors_sc = [(100/255, 197/255, 235/255, 0.7)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, lda_csv_path, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=ids_stg)

# Example CI connectome for Main Fig. 2
seed_cell_ids = ['576460752680588674'] 
ids_stg='576460752680588674_lda'
plot_stg='ic_lda_022825'
colors_sc = [(232/255, 77/255, 138/255, 0.7)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, lda_csv_path, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=ids_stg)

# Several CIs connectome for  Extended Data Fig. 7
seed_cell_ids = ['576460752680588674', '576460752680445826'] 
ids_stg='ics_lda'
plot_stg='ics_lda_022825'
colors_sc = [(232/255, 77/255, 138/255, 0.7)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, lda_csv_path, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=ids_stg)

# Example MC connectome for Main Fig. 2
seed_cell_ids = ['576460752684182585']  
ids_stg='576460752684182585_lda'
plot_stg='mc_lda_022825'
colors_sc = [(127/255, 88/255, 175/255, 0.7)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, lda_csv_path, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=ids_stg)

# Example II connectome for Main Fig. 2 
seed_cell_ids = ['576460752631366630'] 
ids_stg='576460752631366630_lda'
plot_stg='ii_lda_022825'
colors_sc = [(254/255, 179/255, 38/255)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, lda_csv_path, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=ids_stg)

# Example II connectome for Main Fig. 2 
seed_cell_ids = ['576460752671949300'] 
ids_stg='576460752671949300_lda'
plot_stg='ii_3_lda_022825'
colors_sc = [(254/255, 179/255, 38/255)] * 3 * len(seed_cell_ids)
output_neurons, input_neurons, output_seed_counter, input_seed_counter = get_outputs_inputs_neurons(ROOT_FOLDER, lda_csv_path, seed_cell_ids, OUTPUT_PATH_NUCLEI_IDs, seed_id_string=ids_stg)
# Remove the 'noisy neurons'
output_neurons = output_neurons[output_neurons['functional classifier'] != 'noisy, little modulation']
input_neurons = input_neurons[input_neurons['functional classifier'] != 'noisy, little modulation']

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Drop duplicates
output_neurons = output_neurons.drop_duplicates(subset=['axon_id'])
input_neurons = input_neurons.drop_duplicates(subset=['axon_id'])

# Load meshes 
axon_sc, dendrite_sc, soma_sc = find_and_load_cell_meshes(ROOT_FOLDER, seed_cell_ids)
mesh_sc = axon_sc + dendrite_sc + soma_sc

# Get background brain region meshes 
which_brs='raphe'
brain_meshes = load_brs(path_to_data, which_brs=which_brs)
color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)

# Output connectome 
meshes_rec=[]; meshes_notrec=[]; 
colors_rec=[]; colors_notrec=[]; 

for cell_idx in range(len(output_neurons)):
    # Load meshes
    cell_id = output_neurons.iloc[cell_idx, 5]

    # Not functionally recorded 
    if output_neurons.iloc[cell_idx, 1] == 'not functionally imaged':  

        # Myelinated neuron
        if output_neurons.iloc[cell_idx, 9]=="myelinated": 
                
            axon_mesh, dendrite_mesh, soma_mesh=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])

            colors_soma = [COLOR_CELL_TYPE_DICT.get("myelinated")]
            colors_dendrites = [COLOR_CELL_TYPE_DICT.get("myelinated")]
            colors_axon = [COLOR_CELL_TYPE_DICT.get("myelinated")]

            # Merge all recorded meshes and colors
            meshes_rec = meshes_rec + soma_mesh + dendrite_mesh + axon_mesh
            colors_rec = colors_rec + colors_soma + colors_dendrites + colors_axon

        else: 
            axon_mesh_notrec, dendrite_mesh_notrec, soma_mesh_notrec=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])
            colors_soma_notrec = [(0, 0, 0, 0.2)]
            colors_dendrites_notrec = [(0, 0, 0, 0.2)]
            colors_axon_notrec = [(0, 0, 0, 0.2)]

            # Merge all recorded meshes and colors
            meshes_notrec = meshes_notrec + soma_mesh_notrec + dendrite_mesh_notrec + axon_mesh_notrec
            colors_notrec = colors_notrec + colors_soma_notrec + colors_dendrites_notrec + colors_dendrites_notrec
       
    # Recorded
    elif output_neurons.iloc[cell_idx, 1] != 'not functionally imaged': 
        axon_mesh, dendrite_mesh, soma_mesh=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])

        if output_neurons.iloc[cell_idx, 9]=='integrator' and output_neurons.iloc[cell_idx, 11]=='ipsilateral': 
            colors_soma = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]
            colors_dendrites = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]
            colors_axon = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]

        elif output_neurons.iloc[cell_idx, 9]=='integrator' and output_neurons.iloc[cell_idx, 11]=='contralateral': 
            colors_soma = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]
            colors_dendrites = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]
            colors_axon = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]      

        elif output_neurons.iloc[cell_idx, 9]=='dynamic_threshold': 
            colors_soma = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]
            colors_dendrites = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]
            colors_axon = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]

        elif output_neurons.iloc[cell_idx, 9]=='motor_command': 
            colors_soma = [COLOR_CELL_TYPE_DICT.get("motor_command")]
            colors_dendrites = [COLOR_CELL_TYPE_DICT.get("motor_command")]
            colors_axon = [COLOR_CELL_TYPE_DICT.get("motor_command")]
        else: 
            colors_soma = [(0, 0, 0, 0.2)]
            colors_dendrites = [(0, 0, 0, 0.2)]
            colors_axon = [(0, 0, 0, 0.2)]

        # Merge all recorded meshes and colors
        meshes_rec = meshes_rec + soma_mesh + dendrite_mesh + axon_mesh
        colors_rec = colors_rec + colors_soma + colors_dendrites + colors_axon

# Make output connectome plot 
projection='y'
plot_output_connectome(projection, brain_meshes, color_meshes, meshes_notrec, colors_notrec, meshes_rec+mesh_sc, colors_rec+colors_sc, which_brs, path_to_data, plot_stg)

projection='z'
plot_output_connectome(projection, brain_meshes, color_meshes, meshes_notrec, colors_notrec, meshes_rec+mesh_sc, colors_rec+colors_sc, which_brs, path_to_data, plot_stg)
 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input connectome 
meshes_rec=[]; meshes_notrec=[]; meshes_axons=[]; 
colors_rec=[]; colors_notrec=[]; colors_axons=[]; 

for cell_idx in range(len(input_neurons)):

    # Case axon caudal 
    if input_neurons.iloc[cell_idx, 0] == 'axon' and 'caudal' in input_neurons.iloc[cell_idx, 3]: 

        axon_id=input_neurons.iloc[cell_idx, 7]
        axon_mesh = find_and_load_cell_meshes(ROOT_FOLDER, [str(axon_id)])

        meshes_axons = meshes_axons + list(axon_mesh)
        colors_axons = colors_axons + [COLOR_CELL_TYPE_DICT.get("axon")]

    # Case axon rostral
    elif input_neurons.iloc[cell_idx, 0] == 'axon' and 'rostral' in input_neurons.iloc[cell_idx, 3]: 

        axon_id=input_neurons.iloc[cell_idx, 7]
        axon_mesh = find_and_load_cell_meshes(ROOT_FOLDER, [str(axon_id)])

        meshes_axons = meshes_axons + list(axon_mesh)
        colors_axons = colors_axons + [COLOR_CELL_TYPE_DICT.get("axon")]

    # Case neuron
    elif input_neurons.iloc[cell_idx, 0] == 'cell': 

         # Load meshes
        cell_id=input_neurons.iloc[cell_idx, 5]

        # Not functionally recorded 
        if input_neurons.iloc[cell_idx, 1] == 'not functionally imaged':  

            # Myelinated neuron
            if input_neurons.iloc[cell_idx, 9]=="myelinated": 
                    
                axon_mesh, dendrite_mesh, soma_mesh=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])

                colors_soma = [COLOR_CELL_TYPE_DICT.get("myelinated")]
                colors_dendrites = [COLOR_CELL_TYPE_DICT.get("myelinated")]
                colors_axon = [COLOR_CELL_TYPE_DICT.get("myelinated")]

                # Merge all recorded meshes and colors
                meshes_notrec = meshes_notrec + soma_mesh + dendrite_mesh + axon_mesh
                colors_notrec = colors_notrec + colors_soma + colors_dendrites + colors_axon

            else: 
                axon_mesh_notrec, dendrite_mesh_notrec, soma_mesh_notrec=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])
                colors_soma_notrec = [(0, 0, 0, 0.2)]
                colors_dendrites_notrec = [(0, 0, 0, 0.2)]
                colors_axon_notrec = [(0, 0, 0, 0.2)]

                # Merge all recorded meshes and colors
                meshes_notrec = meshes_notrec + soma_mesh_notrec + dendrite_mesh_notrec + axon_mesh_notrec
                colors_notrec = colors_notrec + colors_soma_notrec + colors_dendrites_notrec + colors_dendrites_notrec
        
        # Recorded
        elif input_neurons.iloc[cell_idx, 1] != 'not functionally imaged': 
            axon_mesh, dendrite_mesh, soma_mesh=find_and_load_cell_meshes(ROOT_FOLDER, [str(cell_id)])

            if input_neurons.iloc[cell_idx, 9]=='integrator' and input_neurons.iloc[cell_idx, 11]=='ipsilateral': 
                colors_soma = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]
                colors_dendrites = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]
                colors_axon = [COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")]

            elif input_neurons.iloc[cell_idx, 9]=='integrator' and input_neurons.iloc[cell_idx, 11]=='contralateral': 
                colors_soma = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]
                colors_dendrites = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]
                colors_axon = [COLOR_CELL_TYPE_DICT.get("integrator_contralateral")]      

            elif input_neurons.iloc[cell_idx, 9]=='dynamic_threshold': 
                colors_soma = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]
                colors_dendrites = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]
                colors_axon = [COLOR_CELL_TYPE_DICT.get("dynamic_threshold")]

            elif input_neurons.iloc[cell_idx, 9]=='motor_command': 
                colors_soma = [COLOR_CELL_TYPE_DICT.get("motor_command")]
                colors_dendrites = [COLOR_CELL_TYPE_DICT.get("motor_command")]
                colors_axon = [COLOR_CELL_TYPE_DICT.get("motor_command")]
            else: 
                colors_soma = [(0, 0, 0, 0.2)]
                colors_dendrites = [(0, 0, 0, 0.2)]
                colors_axon = [(0, 0, 0, 0.2)]

            # Merge all recorded meshes and colors
            meshes_rec = meshes_rec + soma_mesh + dendrite_mesh + axon_mesh
            colors_rec = colors_rec + colors_soma + colors_dendrites + colors_axon
        
# Make input connectome plot 
projection='y'
plot_input_connectome(projection, brain_meshes, color_meshes, meshes_axons, colors_axons, 
                      meshes_notrec, colors_notrec, meshes_rec+mesh_sc, colors_rec+colors_sc, which_brs, path_to_data, plot_stg)
projection='z'
plot_input_connectome(projection, brain_meshes, color_meshes, meshes_axons, colors_axons, 
                      meshes_notrec, colors_notrec, meshes_rec+mesh_sc, colors_rec+colors_sc, which_brs, path_to_data, plot_stg)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 10: Regenerate enhanced neural network 

# Load enhanced data
all_cells_xls_hemisphere_lda = pd.read_csv(lda_csv_path)

# All IDs 
dt_ids_all_nuc_lda, dt_ids_all_fun_lad = fetch_filtered_ids(all_cells_xls_hemisphere_lda, 9, 'dynamic_threshold'); 
ic_ids_all_nuc_lda,  ic_ids_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere_lda, 9, 'integrator', 11, 'contralateral')
ii_ids_all_nuc_lda, ii_ids_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere_lda, 9, 'integrator', 11, 'ipsilateral')
mc_ids_all_nuc_lda, mc_ids_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere_lda, 9, 'motor_command')

# Get connectome 
connectome_lda = get_inputs_outputs_by_hemisphere(
    root_folder=ROOT_FOLDER,
    seed_cell_ids=ic_ids_all_nuc_lda,
    hemisphere_df=all_cells_xls_hemisphere_lda
)

# Compute probabilities 
probabilities_results_lda = compute_count_probabilities_from_results(connectome_lda, functional_only=False)
same_side_outputs_cells_cat = probabilities_results_lda["outputs"]["same_side"]["cells"]
same_side_outputs_synapses_cat = probabilities_results_lda["outputs"]["same_side"]["synapses"]
same_side_inputs_cells_cat = probabilities_results_lda["inputs"]["same_side"]["cells"]
same_side_inputs_synapses_cat = probabilities_results_lda["inputs"]["same_side"]["synapses"]

different_side_outputs_cells_cat = probabilities_results_lda["outputs"]["different_side"]["cells"]
different_side_outputs_synapses_cat = probabilities_results_lda["outputs"]["different_side"]["synapses"]
different_side_inputs_cells_cat = probabilities_results_lda["inputs"]["different_side"]["cells"]
different_side_inputs_synapses_cat = probabilities_results_lda["inputs"]["different_side"]["synapses"]

sum_outputs_count = same_side_outputs_synapses_cat['Count'].sum() + different_side_outputs_synapses_cat['Count'].sum()
print("Total output synapses:", sum_outputs_count)

sum_outputs_count = same_side_inputs_synapses_cat['Count'].sum() + different_side_inputs_synapses_cat['Count'].sum()
print("Total input synapses:", sum_outputs_count)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # Adjusted figsize for compactness

# Titles for each subplot
titles = [
    "Same Side Inputs Synapses",
    "Different Side Inputs Synapses",
    "Same Side Outputs Synapses",
    "Different Side Outputs Synapses"
]

# DataFrames corresponding to each subplot
dataframes = [
    same_side_inputs_synapses_cat,
    different_side_inputs_synapses_cat,
    same_side_outputs_synapses_cat,
    different_side_outputs_synapses_cat
]

# Connection type for each DataFrame
connection_types = [
    "inputs",  # Same Side Inputs Synapses
    "inputs",  # Different Side Inputs Synapses
    "outputs",  # Same Side Outputs Synapses
    "outputs"  # Different Side Outputs Synapses
]

# Whether to show the midline based on the subplot's context
show_midlines = [
    False,  # Same Side Inputs Synapses
    True,   # Different Side Inputs Synapses
    False,  # Same Side Outputs Synapses
    True    # Different Side Outputs Synapses
]

# Loop through subplots and draw the network for each
for ax, title, data_df, connection_type, show_midline in zip(axes.flatten(), titles, dataframes, connection_types, show_midlines):
    draw_two_layer_neural_net(
        ax=ax,
        left=0.1, right=0.6, bottom=0.5, top=1.1,
        data_df=data_df,
        node_radius=0.02,  # Larger node size
        input_circle_color='integrator_contralateral',  # Input node color #integrator_contralateral
        input_cell_type='inhibitory',  # Excitatory connections with arrows
        show_midline=show_midline,  # Show the midline only for "different side" plots
        proportional_lines=True,  # Proportional connection line thickness
        a=6, b=2,  # Scale factors for line and arrow sizes
        connection_type=connection_type  # Set the connection type (inputs or outputs)
    )
    ax.set_title(title, fontsize=14)

# Adjust spacing between subplots for compactness
plt.tight_layout()

# Show and save the figure
plot_str = "ic_all_lda_022825"
output_pdf_path = os.path.join(OUTPUT_PATH_NETWORKS, f"neural_network_visualization_with_lda_{plot_str}.pdf")
plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
plt.show()
print(f"Visualization saved as: {output_pdf_path}")
