import datetime
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import navis
import numpy as np
import pandas as pd
import trimesh as tm
from scipy.signal import savgol_filter


def check_registered_swc(file_name):  # drop rows where swc file is not registered
    if Path(rf"W:\Florian\function-neurotransmitter-morphology\{file_name}\{file_name}-000_registered.swc").exists():
        return file_name
    else:
        return np.nan


def fix_duplicates(df):  # modfiy swcs so they dont make trouble when creating obj
    import copy

    work_df = copy.deepcopy(df)

    adoptive_parent = {}
    drop_set = set()
    drop_not_set = set()

    for i_outer, item_outer in work_df.iterrows():

        for i, item in work_df.iterrows():
            if item['node_id'] - 1 not in drop_not_set:
                if (item['x'] == item_outer['x']) and (item['y'] == item_outer['y']) and (item['z'] == item_outer['z']) and (item['node_id'] != item_outer['node_id']):
                    if item_outer['node_id'] in adoptive_parent.keys():
                        while item_outer['node_id'] in adoptive_parent.keys():
                            item_outer['node_id'] = adoptive_parent[item_outer['node_id']]
                    else:
                        adoptive_parent[int(item["node_id"])] = int(item_outer['node_id'])
                    drop_set.add(int(item["node_id"] - 1))
                    drop_not_set.add(int(item_outer['node_id'] - 1))

    print("DROP SET", list(drop_set))

    work_df.drop(list(drop_set), axis=0, inplace=True)
    work_df.loc[:, 'parent_id'].replace(to_replace=adoptive_parent, inplace=True)

    for i, item in work_df.iterrows():
        if item['parent_id'] in adoptive_parent.keys():
            work_df.loc[i, 'parent_id'] = adoptive_parent[item['parent_id']]

    return work_df


def nid(input):  # write neurotransmitter
    if input == '-':
        return 'gad1b'
    if input == '+':
        return 'vglut2a'
    if input == "":
        return 'to be identified '

    else:
        return 'na'


def format_dict(d):
    lines = []
    for key, value in d.items():
        if isinstance(value, list):
            value = ', '.join(f"'{item}'" for item in value)
            lines.append(f"{key} = [{value}]")
        else:
            lines.append(f"{key} = '{value}'")
    return '\n'.join(lines)


def extract_and_df_f(functional_name, volume_name):
    try:
        destination_hdf5_path = rf"W:\Florian\function-neurotransmitter-morphology\export\{volume_name}\{volume_name}_dynamics.hdf5"
        with h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\functional\{functional_name}\{functional_name}_preprocessed_data.h5") as f:
            data_stim0 = np.array(f[f'z_plane0000/manual_segmentation/stimulus_aligned_dynamics/stimulus0000/F'])
            data_stim1 = np.array(f[f'z_plane0000/manual_segmentation/stimulus_aligned_dynamics/stimulus0001/F'])

        with h5py.File(destination_hdf5_path, "w") as f:

            # # raw dynamics
            # f.create_dataset('raw/single_trial_rdms_left', data=data_stim0, dtype=data_stim0.dtype)
            # f.create_dataset('raw/single_trial_rdms_right', data=data_stim1, dtype=data_stim1.dtype)
            # f.create_dataset('raw/average_rdms_left', data=np.nanmean(data_stim0, axis=0).flatten(), dtype=data_stim0.dtype)
            # f.create_dataset('raw/average_rdms_right', data=np.nanmean(data_stim1, axis=0).flatten(), dtype=data_stim1.dtype)


            # #normalized raw dynamics
            # f.create_dataset('normalized/single_trial_rdms_left', data=data_stim0/np.max(data_stim0,axis=2)[:,:,np.newaxis], dtype=data_stim0.dtype)
            # f.create_dataset('normalized/single_trial_rdms_right', data=data_stim1/np.max(data_stim1,axis=2)[:,:,np.newaxis], dtype=data_stim1.dtype)
            # f.create_dataset('normalized/average_rdms_left', data=np.nanmean(data_stim0/np.max(data_stim0,axis=2)[:,:,np.newaxis], axis=0).flatten(), dtype=data_stim0.dtype)
            # f.create_dataset('normalized/average_rdms_right', data=np.nanmean(data_stim1/np.max(data_stim1,axis=2)[:,:,np.newaxis], axis=0).flatten(), dtype=data_stim1.dtype)


            #calculation dF/F on not normalized data of single trials
            stim0_st_dF_F = ((data_stim0 - np.nanmean(data_stim0[:, :, 0:20], axis=2)[:, :, np.newaxis]) / np.nanmean(data_stim0[:, :, 0:20], axis=2)[:, :, np.newaxis])
            stim1_st_dF_F = ((data_stim1 - np.nanmean(data_stim1[:, :, 0:20], axis=2)[:, :, np.newaxis]) / np.nanmean(data_stim1[:, :, 0:20], axis=2)[:, :, np.newaxis])

            # calculation dF/F on not normalized data of avg trials
            # stim0_avg_dF_F = (np.nanmean(data_stim0, axis=0).flatten() - np.nanmean(np.nanmean(data_stim0, axis=0).flatten()[0:20])) / np.nanmean(np.nanmean(data_stim0, axis=0).flatten()[0:20])
            # stim1_avg_dF_F = (np.nanmean(data_stim1, axis=0).flatten() - np.nanmean(np.nanmean(data_stim1, axis=0).flatten()[0:20]) / np.nanmean(np.nanmean(data_stim1, axis=0).flatten()[0:20]))


            #calculation like jon
            F_left_dots = data_stim0
            F_right_dots = data_stim1
            dt = 0.5
            # Compute deltaF/F for each trial, for the 4 tested stimuli
            F0_left_dots = np.nanmean(data_stim0[:, :, int(0 / dt):int(10 / dt)], axis=2, keepdims=True)
            # F0_left_dots = np.nanmean(data_stim0[:,:, int(5/dt):int(10/dt)])
            df_F_left_dots = 100 * (F_left_dots - F0_left_dots) / F0_left_dots
            F0_right_dots = np.nanmean(F_right_dots[:, :, int(0 / dt):int(10 / dt)], axis=2, keepdims=True)
            # F0_right_dots = np.nanmean(data_stim1[:,:, int(5/dt):int(10/dt)])

            df_F_right_dots = 100 * (F_right_dots - F0_right_dots) / F0_right_dots
            # Average over trials
            avg_df_F_left_dots = np.nanmean(df_F_left_dots, axis=0)
            avg_df_F_right_dots = np.nanmean(df_F_right_dots, axis=0)



            #create the datasets for dF/F
            f.create_dataset('dF_F/single_trial_rdms_left', data=df_F_left_dots, dtype=df_F_left_dots.dtype)
            f.create_dataset('dF_F/single_trial_rdms_right', data=df_F_right_dots, dtype=df_F_right_dots.dtype)

            f.create_dataset('dF_F/average_rdms_left_dF_F_calculated_on_single_trials', data=avg_df_F_left_dots.flatten(), dtype=avg_df_F_left_dots.flatten().dtype)
            f.create_dataset('dF_F/average_rdms_right_dF_F_calculated_on_single_trials', data=avg_df_F_right_dots.flatten(), dtype=avg_df_F_right_dots.flatten().dtype)

            # f.create_dataset('dF_F/average_rdms_left_dF_F_calculated_on_average', data=stim0_avg_dF_F, dtype=stim0_st_dF_F.dtype)
            # f.create_dataset('dF_F/average_rdms_right_dF_F_calculated_on_average', data=stim1_avg_dF_F.flatten(), dtype=stim1_st_dF_F.dtype)

            #select dynamics
            single_trials_left = df_F_left_dots
            single_trials_right = df_F_right_dots
            average_left = np.nanmean(single_trials_left,axis=0)
            average_right = np.nanmean(single_trials_right,axis=0)

            smooth_avg_activity_left = savgol_filter(average_left, 20, 3)
            smooth_avg_activity_right = savgol_filter(average_right, 20, 3)
            smooth_trials_left = savgol_filter(single_trials_left[:,0,:], 20, 3, axis=1)
            smooth_trials_right = savgol_filter(single_trials_right[:,0,:], 20, 3, axis=1)

            # Define time axis in seconds
            dt = 0.5  # Time step is 0.5 seconds
            time_axis = np.arange(len(smooth_avg_activity_left.flatten())) * dt

            # Plot smoothed average activity with thin lines
            fig, ax = plt.subplots()
            plt.plot(time_axis, smooth_avg_activity_left[0], color="blue", alpha=0.7, linewidth=3, label="Smoothed Average Left")
            plt.plot(time_axis, smooth_avg_activity_right[0], color="red", alpha=0.7, linestyle="--", linewidth=3, label="Smoothed Average Right")
            # Plot individual trial data with thin black lines for left and dashed black lines for right
            for trial_left, trial_right in zip(smooth_trials_left, smooth_trials_right):
                # if np.max(trial_left) <= 300:
                plt.plot(time_axis, trial_left, color="black", alpha=0.3, linewidth=1)
                # if np.min(trial_right) >= -300:
                plt.plot(time_axis, trial_right, color="black", alpha=0.3, linestyle="--", linewidth=1)
            # Overlay shaded rectangle for stimulus epoch
            plt.axvspan(10, 50, color="gray", alpha=0.1, label="Stimulus Epoch")
            plt.title(f"Average and Individual Trial Activity Dynamics\nNeuron {volume_name}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Activity")
            # Set font of legend text to Arial
            legend = plt.legend()
            for text in legend.get_texts():
                text.set_fontfamily("Arial")
            # Set aspect ratio to 1
            ratio = 1.0
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
            plt.savefig(rf"W:\Florian\function-neurotransmitter-morphology\export\{volume_name}\{volume_name}_rdms.png")
            os.makedirs(r'W:\Florian\function-neurotransmitter-morphology\functional\all_dynamics',exist_ok=True)
            plt.savefig(rf"W:\Florian\function-neurotransmitter-morphology\functional\all_dynamics\{volume_name}_rdms.png")
            plt.show()




            destination_hdf5_path = rf"W:\Florian\function-neurotransmitter-morphology\export\{volume_name}\{volume_name}_dynamics.hdf5"
    except Exception as e:
        print(f"ERROR while processing FUNCTIONAL\n{destination_hdf5_path}\n{e}\n")


# prepare the cell register
cell_register = pd.read_csv(r"C:\Users\ag-bahl\Downloads\cell_register.csv")  # load data
cell_register = cell_register[cell_register['USEABLE'] == 'Y']  # drop empty rows

for i, item in cell_register.iterrows():
    valid_files = ''
    if ' ' in item['Function'].rstrip():
        for functional_name in item.Function.split(' '):
            temp_path = fr'W:\Florian\function-neurotransmitter-morphology\functional\{functional_name}\{functional_name}_roi.tiff'
            if Path(temp_path).exists() and valid_files == '':
                valid_files += functional_name
            elif Path(temp_path).exists():
                valid_files = valid_files + " " + functional_name
        cell_register.loc[cell_register['Internal name'] == item["Internal name"], 'Functional'] = valid_files
    else:
        temp_path = fr'W:\Florian\function-neurotransmitter-morphology\functional\{item.Function}\{item.Function}_roi.tiff'
        if Path(temp_path).exists():
            valid_files += item.Function

    cell_register.loc[cell_register['Internal name'] == item["Internal name"], 'Functional'] = valid_files

cell_register = cell_register[cell_register['Functional'].isna() | ~(cell_register['Functional'] == '')]  # drop empty rows

# drop cells that are not registered
cell_register['Volume'] = cell_register['Volume'].apply(check_registered_swc)
cell_register = cell_register[~cell_register['Volume'].isna()]

# loop through neurons


for i, item in cell_register.iterrows():
    extract_and_df_f(item['Functional'], item["Volume"])
    neuron = navis.read_swc(Path(rf"W:\Florian\function-neurotransmitter-morphology\{item['Volume']}\{item['Volume']}-000_registered.swc"))
    neuron.soma = 1
    neuron.nodes.iloc[0, 5] = 2
    # metadatea
    id = item['Internal name']
    name = f"{item['Volume']}_mapped.swc"
    units = 'microns'
    tracer_name = 'Florian Kaempf'
    classifier = item['Manually evaluated cell type'] + ", " + nid(item["NID"])
    certainty_NID = str(item['certainty of NID assessment'])
    imaging_modality = 'LM - photoactivation'
    date_of_the_tracing = datetime.datetime.fromtimestamp(os.path.getmtime(Path(rf"W:\Florian\function-neurotransmitter-morphology\{item['Volume']}\{item['Volume']}-000_registered.swc"))).strftime("%Y-%m-%d_%H-%M-%S")
    soma_position = f'{neuron.nodes.loc[0, "x"]},{neuron.nodes.loc[0, "y"]},{neuron.nodes.loc[0, "z"]}'

    my_meta = {'id': str(id), 'name': name, 'units': units, 'tracer_name': tracer_name, 'classifier': classifier, "certainty_NID": certainty_NID, 'imaging_modality': imaging_modality,
               'date_of_the_tracing': date_of_the_tracing, 'soma_position': soma_position}

    swc = neuron.nodes
    swc['label'] = swc['label'].astype('int')
    swc.loc[swc['type'] == 'slab', 'label'] = 1
    swc.loc[swc['type'] == 'root', 'label'] = 1
    swc.loc[swc['type'] == 'branch', 'label'] = 5
    swc.loc[swc['type'] == 'end', 'label'] = 6
    swc.drop('type', axis=1, inplace=True)

    # write metadata to swc
    neuron = navis.read_swc(Path(rf"W:\Florian\function-neurotransmitter-morphology\{item['Volume']}\{item['Volume']}-000_registered.swc"), id=str(id), name=name, units=units, tracer_name=tracer_name,
                            classifier=classifier, certainty_NID=certainty_NID, imaging_modality=imaging_modality, date_of_the_tracing=date_of_the_tracing, soma_position=soma_position)

    if not Path(rf"W:\Florian\function-neurotransmitter-morphology\export\{item['Volume']}\{item['Volume']}_mapped.swc").exists():
        swc = fix_duplicates(swc)

    neuron.nodes = swc
    neuron.soma = 1
    neuron.nodes.loc[:, 'radius'] = 0.3
    # neuron.nodes.loc[1:, 'radius'] = 0.3
    # neuron.nodes.iloc[0, 5] = 2
    neuron_mesh = navis.conversion.tree2meshneuron(neuron, use_normals=True, tube_points=20)

    os.makedirs(rf"W:\Florian\function-neurotransmitter-morphology\export\{item['Volume']}", exist_ok=True)
    # swc write
    neuron.to_swc(Path(rf"W:\Florian\function-neurotransmitter-morphology\{item['Volume']}\{item['Volume']}-000_registered_metadata.swc"), write_meta=my_meta)
    neuron.to_swc(Path(rf"W:\Florian\function-neurotransmitter-morphology\all_swc\{item['Volume']}-000_registered_metadata.swc"), write_meta=my_meta)
    neuron.to_swc(Path(rf"W:\Florian\function-neurotransmitter-morphology\export\{item['Volume']}\{item['Volume']}_mapped.swc"), write_meta=my_meta)

    # obj write
    navis.write_mesh(neuron_mesh, rf"W:\Florian\function-neurotransmitter-morphology\all_obj\{item['Volume']}-000_registered_metadata.obj")
    navis.write_mesh(neuron_mesh, rf"W:\Florian\function-neurotransmitter-morphology\{item['Volume']}\{item['Volume']}-000_registered_metadata.obj")
    navis.write_mesh(neuron_mesh, rf"W:\Florian\function-neurotransmitter-morphology\export\{item['Volume']}\{item['Volume']}_mapped.obj")

    # create soma obj

    x = neuron.nodes.loc[0, "x"]
    y = neuron.nodes.loc[0, "y"]
    z = neuron.nodes.loc[0, "z"]

    sphere = tm.creation.icosphere(radius=2, subdivisions=2)
    sphere.apply_translation((x, y, z))
    sphere.export(rf"W:\Florian\function-neurotransmitter-morphology\export\{item['Volume']}\{item['Volume']}_mapped_soma.obj")

    # create fused mesh file with soma and neurites
    neurites = tm.load_mesh(rf"W:\Florian\function-neurotransmitter-morphology\export\{item['Volume']}\{item['Volume']}_mapped.obj")
    soma = tm.load_mesh(rf"W:\Florian\function-neurotransmitter-morphology\export\{item['Volume']}\{item['Volume']}_mapped_soma.obj")
    combined_list = [neurites, soma]
    scene = tm.Scene(combined_list)
    scene.export(rf"W:\Florian\function-neurotransmitter-morphology\export\{item['Volume']}\{item['Volume']}_mapped_combined.obj")

    # write metadata to txt

    with open(rf"W:\Florian\function-neurotransmitter-morphology\export\{item['Volume']}\metadata.txt", 'w') as convert_file:
        convert_file.write(format_dict(my_meta))
