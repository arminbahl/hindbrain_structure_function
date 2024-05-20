
# This codes pulls segments and synapses from Neuroglancer for clem_zfish1
# It requires a private Cloudvolume token to obtain it contact jonathan.boulanger.weill@gmail.com
# Compatible with Python 3.10.13
# Install environnement using conda env create --file pull_from_neuroglancer.yaml
# Version: 0.4 / 05/20/2024 jbw

############################################################################################################################
############################################################################################################################
import navis
import cloudvolume as cv
import numpy as np
import os
from pathlib import Path
import pandas as pd
from pandas.api.types import is_string_dtype
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib

navis.patch_cloudvolume()
# Get Graphene token is required
vol = cv.CloudVolume(
    "graphene://https://data.proofreading.zetta.ai/segmentation/api/v1/lichtman_zebrafish_hindbrain_001",
    use_https=True,
    progress=False,
)

from caveclient import CAVEclient
client = CAVEclient(
    datastack_name="lichtman_zebrafish_hindbrain",
    server_address="https://proofreading.zetta.ai",
)
import datetime

############################################################################################################################
############################################################################################################################

# Generate the metadata files and import objs. 

# Load the entire workbook.
df = pd.read_excel(
    "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/all_cells/all_cells.xlsx"
)
num_cells = df.shape

# Root path to export all_cells data
root_path = Path(
    "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/all_cells/"
)

# Test if the axon and dendrite segments are up to date
synapse_table = client.info.get_datastack_info()['synapse_table']

problematic_axons = []
for element_idx in range(0, num_cells[0]): 
    axon_id = str(df.iloc[element_idx, 5])
    try:
        # Your code that might raise an exception
        output_synapses=client.materialize.live_query(synapse_table,
                                            datetime.datetime.now(datetime.timezone.utc),
                                            filter_equal_dict = {'pre_pt_root_id': int(axon_id)})
    except Exception as e:
        # If an exception occurs, store the problematic value
        problematic_axons.append((axon_id))
        continue  # Continue with the next iteration
problematic_dendrites = []
for element_idx in range(0, num_cells[0]):
        dendrites_id = str(df.iloc[element_idx, 6])
        if dendrites_id != 0: #in case we are only downloading an axon
            try:
            # Your code that might raise an exception
                input_synapses=client.materialize.live_query(synapse_table,
                                            datetime.datetime.now(datetime.timezone.utc),
                                            filter_equal_dict = {'post_pt_root_id': int(dendrites_id)}) 
            except Exception as e:
                # If an exception occurs, store the problematic value
                problematic_dendrites.append((dendrites_id))
        
# After the loop, you can examine the problematic values
print("Problematic axons:", problematic_axons)
print("Problematic dendrites:", problematic_dendrites)

# Generate metadata 
for element_idx in range(0, num_cells[0]):
    print(element_idx)

    # Cell or axon 
    element_type=str(df.iloc[element_idx, 0])

    # Case element is a cell 
    if element_type=='cell':
        functional_id = str(df.iloc[element_idx, 1])
        # case when no comments 
        if pd.isnull(df.iloc[element_idx, 2]):
            comment = "na"
        else:
            comment = str(df.iloc[element_idx, 2])
        nucleus_id= str(df.iloc[element_idx, 3])  
        soma_id = str(df.iloc[element_idx, 4])
        axon_id = str(df.iloc[element_idx, 5])
        dendrites_id = str(df.iloc[element_idx, 6])
        segment_id = "clem_zfish1_cell" + "_" + nucleus_id

        functional_classifier = str(df.iloc[element_idx, 7])
        neurotransmitter_classifier = str(df.iloc[element_idx, 8])
        projection_classifier = str(df.iloc[element_idx, 9])
        imaging_modality = str(df.iloc[element_idx, 11])
        date_of_tracing = str(df.iloc[element_idx, 12])
        tracer_names = str(df.iloc[element_idx, 13])
        neuroglancer_link = str(df.iloc[element_idx, 10])

    # Case element is an axon 
    else:
        functional_id = str(df.iloc[element_idx, 1])
        # case when no comments 
        if pd.isnull(df.iloc[element_idx, 2]):
            comment = "na"
        else:
            comment = str(df.iloc[element_idx, 2])
        nucleus_id= "na"
        soma_id = "na"
        axon_id = str(df.iloc[element_idx, 5])
        dendrites_id = "na"
        segment_id = "clem_zfish1_axon" + "_" + axon_id

        functional_classifier = "na"
        neurotransmitter_classifier = "na"
        projection_classifier = "na"
        imaging_modality = str(df.iloc[element_idx, 11])
        date_of_tracing = str(df.iloc[element_idx, 12])
        tracer_names = str(df.iloc[element_idx, 13])
        neuroglancer_link = str(df.iloc[element_idx, 10])

    # Make a directory for the element 
    if not os.path.exists(str(root_path) + "/" + segment_id):
        os.makedirs(str(root_path) + "/" + segment_id)

    # Create the text file containnig the metadata:
    path_text_file = str(root_path) + "/" + segment_id + "/" + segment_id + "_metadata.txt"

    if element_type=='cell':
        lines = [
            "type = \"" + 'cell' + "\"", 
            "cell_name = " + nucleus_id, 
            "nucleus_id = " + nucleus_id, 
            "soma_id = " + soma_id,  
            "axon_id = " + axon_id, 
            "dendrites_id = " + dendrites_id,
            "functional_id = \"" + functional_id + "\"", 
            "cell_type_labels = [" + "\"" + functional_classifier + "\"" + ", " + "\"" + neurotransmitter_classifier + "\"" + ", " + "\"" + projection_classifier + "\"" + "]",
            "imaging_modality = \"" + imaging_modality + "\"",
            "date_of_tracing = " + " " + date_of_tracing,
            "tracer_names = \"" + tracer_names + "\"",
            "neuroglancer_link = \"" + neuroglancer_link + "\"",
            "comment = \"" + comment + "\"",
         ]
    else: 
        lines = [
            "type = \"" + 'axon' + "\"",
            "cell_name = \"" + 'na' + "\"", 
            "nucleus_id = \"" + 'na' + "\"", 
            "soma_id = \"" + 'na' + "\"",  
            "axon_id = " + axon_id, 
            "dendrites_id = \"" + 'na' + "\"",
            "functional_id = \"" + functional_id + "\"", 
            "cell_type_labels = [" + "\"" + functional_classifier + "\"" + ", " + "\"" + neurotransmitter_classifier + "\"" + ", " + "\"" + projection_classifier + "\"" + "]",
            "imaging_modality = \"" + imaging_modality + "\"",
            "date_of_tracing = " + " " + date_of_tracing,
            "tracer_names = \"" + tracer_names + "\"",
            "neuroglancer_link = \"" + neuroglancer_link + "\"",
            "comment = \"" + comment + "\"",
        ]
    with open(path_text_file, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")

    # Upload segments from Neuroglancer
    if element_type=='cell':        
        # Get and save soma and nucleus    
        soma_parts = vol.mesh.get([soma_id, nucleus_id], as_navis=True)
        soma_path = str(root_path) + "/" + segment_id + "/" + segment_id + "_soma.obj"
        soma_nuc = navis.combine_neurons(soma_parts)
        navis.write_mesh(soma_nuc, soma_path, filetype="obj")
        # Get and save axon
        axon = vol.mesh.get([axon_id], as_navis=True)
        axon_path = str(root_path) + "/" + segment_id + "/" + segment_id + "_axon.obj"
        navis.write_mesh(axon, axon_path, filetype="obj")
        # Get and save dendrites
        dendrites = vol.mesh.get([dendrites_id], as_navis=True)
        dendrites_path = str(root_path) + "/" + segment_id + "/" + segment_id + "_dendrite.obj"
        navis.write_mesh(dendrites, dendrites_path, filetype="obj")
        # Get and save the whole neuron
        neuron_parts = vol.mesh.get([soma_id, nucleus_id, axon_id, dendrites_id], as_navis=True)
        neuron_path = str(root_path) + "/" + segment_id + "/" + segment_id + ".obj"
        neuron = navis.combine_neurons(neuron_parts)
        navis.write_mesh(neuron, neuron_path, filetype="obj")
        # Plot to double check
        fig = neuron.plot3d()
        
        # Get the input and output synapses 
        synapse_table = client.info.get_datastack_info()['synapse_table']

        output_synapses=client.materialize.live_query(synapse_table,
                                            datetime.datetime.now(datetime.timezone.utc),
                                            filter_equal_dict = {'pre_pt_root_id': int(axon_id)})
        input_synapses=client.materialize.live_query(synapse_table,
                                    datetime.datetime.now(datetime.timezone.utc),
                                    filter_equal_dict = {'post_pt_root_id': int(dendrites_id)}) 

        output_segment=output_synapses.post_pt_root_id

        output_position=output_synapses.pre_pt_position
        #Synapses are pulled at 16*16*30nm resolution, correct to match to 8*8*30nm
        output_position = output_position.apply(lambda x: [16 * x[0], 16 * x[1], 30 * x[2]])
        output_size=output_synapses.iloc[:, 4]

        input_segment=input_synapses.pre_pt_root_id
        input_position=input_synapses.post_pt_position
        #Synapses are pulled at 8*8*30nm resolution, correct to match to 4*4*30nm
        input_position = input_position.apply(lambda x: [16 * x[0], 16 * x[1], 30 * x[2]])
        input_size=input_synapses.iloc[:, 4]

        # Function to format synapse data
        def format_synapse(segment_id, position, size):
            segment_str = str(segment_id)
            position_str = ','.join(map(str, position))
            return f"{segment_str},{position_str},{size}"

        # Write formatted synapses to file
        path_text_syp_file = str(root_path) + "/" + segment_id + "/" + segment_id + "_synapses.txt"
        print(path_text_syp_file)

        with open(path_text_syp_file, "w") as file:
            file.write("(presynaptic: [")
            for segment, position, size in zip(output_segment, output_position, output_size):
                file.write("'" + format_synapse(segment, position, size) + "', ")   
            file.write("]")       
            file.write(",postsynaptic: [")
            for segment, position, size in zip(input_segment, input_position, input_size):
                file.write("'" + format_synapse(segment, position, size) + "', ") 
            file.write("]") 

        #Activity plots and hdf5 if functionally recorded
        if functional_id != 'not functionally imaged':  

        # Now create an hdf5 file with the activity and plot 
        # Load the h5 file for a speciic neuron
            with h5py.File("/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/function/all_cells.h5", "r") as hdf_file:
                neuron_group = hdf_file[f"neuron_{functional_id}"] 

                # Get average activity for left and right stimuli
                avg_activity_left = neuron_group["average_activity_left"][()]
                avg_activity_right = neuron_group["average_activity_right"][()]
                # Get individual trial data for left and right stimuli
                trials_left = neuron_group["neuronal_activity_trials_left"][()]
                trials_right = neuron_group["neuronal_activity_trials_right"][()]

                # Create a new hdf5 and save 
                destination_hdf5_path= str(root_path) + "/" + segment_id + "/" + segment_id + "_dynamics.hdf5"
                with h5py.File(destination_hdf5_path, "w") as f: 
                    f.create_dataset('dF_F/single_trial_rdms_left', data=avg_activity_left)
                    f.create_dataset('dF_F/single_trial_rdms_right', data=avg_activity_right)
                    f.create_dataset('dF_F/average_rdms_left_dF_F_calculated_on_single_trials', data=avg_activity_left)
                    f.create_dataset('dF_F/average_rdms_right_dF_F_calculated_on_single_trials', data=avg_activity_right)

                # Plot the activity traces
                # Smooth using a Savitzky-Golay filter
                smooth_avg_activity_left = savgol_filter(avg_activity_left, 20, 3)
                smooth_avg_activity_right = savgol_filter(avg_activity_right, 20, 3)
                smooth_trials_left = savgol_filter(trials_left, 20, 3, axis=1)
                smooth_trials_right = savgol_filter(trials_right, 20, 3, axis=1)

                # Define time axis in seconds
                dt = 0.5  # Time step is 0.5 seconds
                time_axis = np.arange(len(avg_activity_left)) * dt

                # Plot smoothed average activity with thin lines
                fig, ax = plt.subplots()
                plt.plot(time_axis, smooth_avg_activity_left, color='blue', alpha=0.7, linewidth=3, label='Smoothed Average Left')
                plt.plot(time_axis, smooth_avg_activity_right, color='red', alpha=0.7, linestyle='--', linewidth=3, label='Smoothed Average Right')
                
                # Plot individual trial data with thin black lines for left and dashed black lines for right
                for trial_left, trial_right in zip(smooth_trials_left, smooth_trials_right):
                    plt.plot(time_axis, trial_left, color='black', alpha=0.3, linewidth=1)
                    plt.plot(time_axis, trial_right, color='black', alpha=0.3, linestyle='--', linewidth=1)
                
                # Overlay shaded rectangle for stimulus epoch
                plt.axvspan(20, 60, color='gray', alpha=0.1, label='Stimulus Epoch')

                plt.title(f'Average and Individual Trial Activity Dynamics for Neuron {functional_id}')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Activity')
                
                # Set font of legend text to Arial
                legend = plt.legend()
                for text in legend.get_texts():
                    text.set_fontfamily('Arial')
                    
                # Set aspect ratio to 1
                ratio = 1.0
                x_left, x_right = ax.get_xlim()
                y_low, y_high = ax.get_ylim()
                ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

                #Save figure 
                path_text_file = str(root_path) + "/" + segment_id + "/" + segment_id + "_dynamics.pdf"
                plt.savefig(path_text_file)
                plt.show()
                
    else: 
        # Get and save axon
        axon = vol.mesh.get([axon_id], as_navis=True)
        axon_path = str(root_path) + "/" + segment_id + "/" + segment_id + "_axon.obj"
        navis.write_mesh(axon, axon_path, filetype="obj")
        
        # Get the input and output synapses 
        synapse_table = client.info.get_datastack_info()['synapse_table']

        output_synapses=client.materialize.live_query(synapse_table,
                                    datetime.datetime.now(datetime.timezone.utc),
                                    filter_equal_dict = {'pre_pt_root_id': int(axon_id)})
        
        output_segment=output_synapses.post_pt_root_id
        output_position=output_synapses.pre_pt_position
        #Synapses are pulled at 16*16*30nm resolution, correct to match to 8*8*30nm
        output_position = output_position.apply(lambda x: [16 * x[0], 16 * x[1], 30 * x[2]])
        output_size=output_synapses.iloc[:, 4]

        # Function to format synapse data
        def format_synapse(segment_id, position, size):
            segment_str = str(segment_id)
            position_str = ','.join(map(str, position))
            return f"{segment_str},{position_str},{size}"

        # Write formatted synapses to file
        path_text_syp_file = str(root_path) + "/" + segment_id + "/" + segment_id + "_synapses.txt"

        with open(path_text_syp_file, "w") as file:
            file.write("(presynaptic: [")
            for segment, position, size in zip(output_segment, output_position, output_size):
                file.write("'" + format_synapse(segment, position, size) + "', ")   
            file.write("]")  
            file.write(",postsynaptic: []")

############################################################################################################################
############################################################################################################################




