# This codes pulls segments from Neuroglancer for clem_zfish1
# It requires a private Cloudvolume token to obtain it contact jonathan.boulanger.weill@gmail.com

# Compatible with Python 3.10.11
# cloud-volume can't be installed direcly within the environment, instead 
# Install cloud-volume 8.30.0 with pip: pip install cloud-volume

# Version: 0.2 / 29/02/2024 jbw

############################################################################################################################
############################################################################################################################
import navis
import cloudvolume as cv
import numpy as np
import skeletor as sk
import trimesh as tm
import os
import subprocess
from pathlib import Path
import pandas as pd

navis.patch_cloudvolume()
# Get Graphene token is required
vol = cv.CloudVolume(
    "graphene://https://data.proofreading.zetta.ai/segmentation/api/v1/lichtman_zebrafish_hindbrain_001",
    use_https=True,
    progress=False,
)

############################################################################################################################
############################################################################################################################

# Generate the metadata files and import objs. 

# Load the entire workbook.
df = pd.read_excel(
    "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/all_cells/all_cells_022824.xlsx"
)
num_cells = df.shape
# Root path to export all_cells data
root_path = Path(
    "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/all_cells/"
)

for cell_idx in range(0, num_cells[0]):
    # Generate metadata 
    nucleus_id= str(df.iloc[cell_idx, 0])
    cell_id = "clem_zfish1" + "_" + nucleus_id
    soma_id = str(df.iloc[cell_idx, 1])
    axon_id = str(df.iloc[cell_idx, 2])
    dendrites_id = str(df.iloc[cell_idx, 3])
    # case when neuron'not funuctionally imaged' 
    if np.isnan(df.iloc[cell_idx, 4]):
        functional_id = "not functionally imaged"
    else:
        functional_id = str(int(df.iloc[cell_idx, 4]))
    functional_classifier = str(df.iloc[cell_idx, 5])
    neurotransmitter_classifier = str(df.iloc[cell_idx, 6])
    projection_classifier = str(df.iloc[cell_idx, 7])
    imaging_modality = str(df.iloc[cell_idx, 8])
    date_of_tracing = str(df.iloc[cell_idx, 9])
    tracer_names = str(df.iloc[cell_idx, 10])
    neuroglancer_link = str(df.iloc[cell_idx, 11])

    # Make a directory for the cell
    if not os.path.exists(str(root_path) + "/" + cell_id):
        os.makedirs(str(root_path) + "/" + cell_id)

    # Create the text file containnig the metadata:
    path_text_file = str(root_path) + "/" + cell_id + "/" + cell_id + "_metadata.txt"
    if np.isnan(df.iloc[cell_idx, 4]): 
        lines = [
            "cell_name = " + nucleus_id, 
            "nucleus_id = " + nucleus_id, 
            "soma_id = " + soma_id,  
            "axon_id = " + axon_id, 
            "dendrites_id = " + dendrites_id,
            "functional_id = \"" + "not functionally imaged" + "\"", 
            "cell_type_labels = [" + "\"" + functional_classifier + "\"" + ", " + "\"" + neurotransmitter_classifier + "\"" + ", " + "\"" + projection_classifier + "\"" + "]",
            "imaging_modality = \"" + imaging_modality + "\"",
            "date_of_tracing = " + " " + date_of_tracing,
            "tracer_names = \"" + tracer_names + "\"",
            "neuroglancer_link = \"" + neuroglancer_link + "\"",
        ]
    else: 
        lines = [
        "cell_name = " + nucleus_id, 
        "nucleus_id = " + nucleus_id, 
        "soma_id = " + soma_id,  
        "axon_id = " + axon_id, 
        "dendrites_id = " + dendrites_id,
        "functional_id = " + functional_id, 
        "cell_type_labels = [" + "\"" + functional_classifier + "\"" + ", " + "\"" + neurotransmitter_classifier + "\"" + ", " + "\"" + projection_classifier + "\"" + "]",
        "imaging_modality = \"" + imaging_modality + "\"",
        "date_of_tracing = " + " " + date_of_tracing,
        "tracer_names = \"" + tracer_names + "\"",
        "neuroglancer_link = \"" + neuroglancer_link + "\"",
    ]
    with open(path_text_file, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")

    # Upload segments from Neuroglancer
    # Get and save soma and nucleus    
    soma_parts = vol.mesh.get([soma_id, nucleus_id], as_navis=True)
    soma_path = str(root_path) + "/" + cell_id + "/" + cell_id + "_soma.obj"
    soma_nuc = navis.combine_neurons(soma_parts)
    navis.write_mesh(soma_nuc, soma_path, filetype="obj")
    # Get and save axon
    axon = vol.mesh.get([axon_id], as_navis=True)
    axon_path = str(root_path) + "/" + cell_id + "/" + cell_id + "_axon.obj"
    navis.write_mesh(axon, axon_path, filetype="obj")
    # Get and save dendrites
    dendrites = vol.mesh.get([dendrites_id], as_navis=True)
    dendrites_path = str(root_path) + "/" + cell_id + "/" + cell_id + "_dendrite.obj"
    navis.write_mesh(dendrites, dendrites_path, filetype="obj")
    # Get and save the whole neuron
    neuron_parts = vol.mesh.get([soma_id, nucleus_id, axon_id, dendrites_id], as_navis=True)
    neuron_path = str(root_path) + "/" + cell_id + "/" + cell_id + ".obj"
    neuron = navis.combine_neurons(neuron_parts)
    navis.write_mesh(neuron, neuron_path, filetype="obj")
    # Plot to double check
    fig = neuron.plot3d()
