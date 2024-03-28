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
import errno
from hindbrain_structure_function.PA.tools_for_export.fix_duplicates import *
from hindbrain_structure_function.PA.tools_for_export.format_dict4metadata import *
from scipy.signal import savgol_filter

#settings
force_new_all = True
use_debug = True
debug_cell = '20230327.1' #["20230324.1","20230327.1"]
fiji_dynamics = True

#fetch user
user = os.getcwd().split('\\')[2]
base_path = Path(rf"C:\Users\{user}\Desktop\hindbrain_structure_function")
base_path_data = Path(rf"W:\Florian\function-neurotransmitter-morphology")

#load dataframe with all cell information

cell_table = pd.read_csv(rf"C:\Users\{user}\Desktop\photoactivation_cells_table.csv")
cell_table = cell_table[(~cell_table['celltype_labels'].isna())&
                        (~cell_table['date_of_tracing'].isna())]
#transform celltype labels to lists in df
for i,cell in cell_table.iterrows():
    cell_table.at[i,'celltype_labels'] = cell['celltype_labels'].replace("[","").replace("]","").replace('"',"").split(',')
#remove all quote chars in strings
for column in cell_table.columns:
    if column != 'celltype_labels':
        cell_table[column] = cell_table[column].apply(lambda x: x.replace('"', '') if type(x) == str else x)

#subset dataframe if in debug mode
if use_debug:
    cell_table = cell_table[cell_table['cellname'] == debug_cell]


#create save space
os.makedirs(base_path,exist_ok=True)
os.makedirs(base_path.joinpath('export_photoactivation'),exist_ok=True)
base_export_folder = base_path.joinpath('export_photoactivation')

#loop over cells
for i,cell in cell_table.iterrows():
    #create home of export file
    if not base_export_folder.joinpath(cell.cellname).exists() or force_new_all or use_debug:
        os.makedirs(base_export_folder.joinpath(cell.cellname),exist_ok=True)
        temp_cell_path = Path(base_export_folder.joinpath(cell.cellname))

        #read swc
        try:
            neuron = navis.read_swc(base_path_data.joinpath(cell.photoactivation_ID).joinpath(cell.photoactivation_ID + "-000_registered.swc"))
        except:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(base_path_data.joinpath(cell.photoactivation_ID).joinpath(cell.photoactivation_ID + "-000_registered.swc")))

        #modfiy neuron from swc
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2

        swc = neuron.nodes
        swc['label'] = swc['label'].astype('int')
        swc['label'] = swc['label'].astype('int')
        swc.loc[swc['type'] == 'slab', 'label'] = 1
        swc.loc[swc['type'] == 'root', 'label'] = 1
        swc.loc[swc['type'] == 'branch', 'label'] = 5
        swc.loc[swc['type'] == 'end', 'label'] = 6
        swc.drop('type', axis=1, inplace=True)

        # write metadata to swc
        neuron = navis.read_swc(base_path_data.joinpath(cell.photoactivation_ID).joinpath(cell.photoactivation_ID + "-000_registered.swc"),
                                cellname=cell['cellname'],
                                celltype_labels=str(cell['celltype_labels']),
                                imaging_modality=cell['imaging_modality'],
                                gad1b_ID=cell['gad1b_ID'],
                                vglut2a_ID=cell['vglut2a_ID'],
                                photoactivation_ID=cell['photoactivation_ID'],
                                verification_ID=cell['verification_ID'],
                                function_ID=cell['function_ID'],
                                date_of_tracing=cell['date_of_tracing'],
                                tracer=cell['tracer'],
                                comments=cell['comments'])

        #remove duplicated nodes and reattaches them for clean generation of obj
        if (swc.loc[:,['x','y','z']]).duplicated().sum() != 0:
            swc = fix_duplicates(swc)

        neuron.nodes = swc
        neuron.soma = 1                     #TODO maybe sync this with jonathan
        neuron.nodes.loc[:, 'radius'] = 0.3 #TODO maybe sync this with jonathan
        neuron_mesh = navis.conversion.tree2meshneuron(neuron, use_normals=True, tube_points=20)

        #export neuron with meta data
        neuron.to_swc(temp_cell_path.joinpath(cell.cellname + '.swc'), write_meta=dict(cell))
        #write obj
        navis.write_mesh(neuron_mesh, temp_cell_path.joinpath(cell.cellname + '.obj'))
        #write a soma obj
        x = neuron.nodes.loc[0, "x"]
        y = neuron.nodes.loc[0, "y"]
        z = neuron.nodes.loc[0, "z"]

        sphere = tm.creation.icosphere(radius=2, subdivisions=2)
        sphere.apply_translation((x, y, z))
        sphere.export(temp_cell_path.joinpath(cell.cellname + '_soma.obj'))

        # create fused mesh file with soma and neurites
        neurites = tm.load_mesh(temp_cell_path.joinpath(cell.cellname + '.obj'))
        soma = tm.load_mesh(temp_cell_path.joinpath(cell.cellname + '_soma.obj'))
        combined_list = [neurites, soma]
        scene = tm.Scene(combined_list)
        scene.export(temp_cell_path.joinpath(cell.cellname + '_combined.obj'))

        # write metadata to txt

        with open(temp_cell_path.joinpath(cell.cellname+'metadata.txt'), 'w') as convert_file:
            convert_file.write(format_dict(dict(cell)))


        #dynamics part


        #there are two weird cells that I extracted manually with fiji
        if cell.cellname in ["20230324.1","20230327.1"] and fiji_dynamics:
            fiji_dynamics_array = np.array(pd.read_excel(base_path_data.joinpath('functional').joinpath(cell.function_ID).joinpath('fiji_dynamics.xlsx')))-10000
            F_left_dots = fiji_dynamics_array[:,0]
            F_right_dots = fiji_dynamics_array[:, 1]

            F_left_dots = (F_left_dots + abs(np.min(F_left_dots[:20], axis=0)))
            F_right_dots = (F_right_dots + abs(np.min(F_right_dots[:20], axis=0)))

            dt=0.5
            # Compute deltaF/F for each trial, for the 4 tested stimuli
            F0_left_dots = np.nanmean(F_left_dots[int(5 / dt):int(10 / dt)], axis=0, keepdims=True)
            df_F_left_dots = 100 * (F_left_dots - F0_left_dots) / F0_left_dots
            F0_right_dots = np.nanmean(F_right_dots[int(5 / dt):int(10 / dt)], axis=0, keepdims=True)
            df_F_right_dots = 100 * (F_right_dots - F0_right_dots) / F0_right_dots
            # Average over trials
            avg_df_F_left_dots = df_F_left_dots
            avg_df_F_right_dots = df_F_right_dots

            with h5py.File(temp_cell_path.joinpath(cell.cellname + '_dynamics.hdf5'), "w") as f:
                f.create_dataset('dF_F/single_trial_dots_left', data=df_F_left_dots, dtype=df_F_left_dots.dtype)
                f.create_dataset('dF_F/single_trial_dots_right', data=df_F_right_dots, dtype=df_F_right_dots.dtype)

                f.create_dataset('dF_F/average_dots_left', data=avg_df_F_left_dots, dtype=avg_df_F_left_dots.dtype)
                f.create_dataset('dF_F/average_dots_right', data=avg_df_F_right_dots, dtype=avg_df_F_right_dots.dtype)

            # smoothing

            smooth_avg_activity_left = savgol_filter(avg_df_F_left_dots, 20, 3)
            smooth_avg_activity_right = savgol_filter(avg_df_F_right_dots, 20, 3)
            smooth_trials_left = savgol_filter(df_F_left_dots, 20, 3)
            smooth_trials_right = savgol_filter(df_F_right_dots, 20, 3)

            # create folder for all dynamics
            os.makedirs(base_export_folder.joinpath('all_dynamics_average'), exist_ok=True)
            os.makedirs(base_export_folder.joinpath('all_dynamics_average_and_single_trials'), exist_ok=True)

            # Define time axis in seconds

            time_axis = np.arange(len(avg_df_F_right_dots)) * dt

            # Plot smoothed average activity
            fig, ax = plt.subplots()
            plt.plot(time_axis, smooth_avg_activity_left, color='blue', alpha=0.7, linewidth=3, label='Smoothed Average Left')
            plt.plot(time_axis, smooth_avg_activity_right, color='red', alpha=0.7, linestyle='--', linewidth=3, label='Smoothed Average Right')
            # Plot individual trial data with thin black lines for left and dashed black lines for right

            # Overlay shaded rectangle for stimulus epoch
            plt.axvspan(10, 50, color='gray', alpha=0.1, label='Stimulus Epoch')
            plt.title(f'Average and Individual Trial Activity Dynamics\nNeuron {cell.cellname}\nDynamics from fiji')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Activity')
            # Set font of legend text to Arial
            legend = plt.legend(frameon=False, loc='upper left', fontsize='small')
            for text in legend.get_texts():
                text.set_fontfamily('Arial')
            # Set aspect ratio to 1
            ratio = 1.0
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            plt.savefig(temp_cell_path.joinpath(cell.cellname+'_rdms_average_fiji.png'))
            plt.savefig(base_export_folder.joinpath('all_dynamics_average').joinpath(cell.cellname + '_rdms_average_fiji.png'))
            plt.savefig(temp_cell_path.joinpath(cell.cellname + '_rdms_average_and_single_trials_fiji.png'))
            plt.savefig(base_export_folder.joinpath('all_dynamics_average_and_single_trials').joinpath(cell.cellname + '_rdms_average_and_single_trials_fiji.png'))
            plt.show()




        else:
            path_functional_data = base_path_data.joinpath('functional').joinpath(cell.function_ID).joinpath(cell.function_ID+"_preprocessed_data.h5")
            with h5py.File(path_functional_data) as f:
                F_left_dots  = np.array(f[f'z_plane0000/manual_segmentation/stimulus_aligned_dynamics/stimulus0000/F']).squeeze()
                F_right_dots = np.array(f[f'z_plane0000/manual_segmentation/stimulus_aligned_dynamics/stimulus0001/F']).squeeze()

            F_left_dots = (F_left_dots + abs(np.nanmean(F_left_dots[:20],axis=0)))
            F_right_dots = (F_right_dots + abs(np.nanmean(F_right_dots[:20],axis=0)))

            dt = 0.5  # Time step is 0.5 seconds

            # Compute deltaF/F for each trial, for the 4 tested stimuli
            F0_left_dots = np.nanmean(F_left_dots[:, int(0 / dt):int(10 / dt)], axis=1, keepdims=True)
            df_F_left_dots = 100 * (F_left_dots - F0_left_dots) / F0_left_dots
            F0_right_dots = np.nanmean(F_right_dots[:, int(0 / dt):int(10 / dt)], axis=1, keepdims=True)
            df_F_right_dots = 100 * (F_right_dots - F0_right_dots) / F0_right_dots
            # Average over trials
            avg_df_F_left_dots = np.nanmean(df_F_left_dots, axis=0)
            avg_df_F_right_dots = np.nanmean(df_F_right_dots, axis=0)

            with h5py.File(temp_cell_path.joinpath(cell.cellname+'_dynamics.hdf5'), "w") as f:
                f.create_dataset('dF_F/single_trial_dots_left', data=df_F_left_dots, dtype=df_F_left_dots.dtype)
                f.create_dataset('dF_F/single_trial_dots_right', data=df_F_right_dots, dtype=df_F_right_dots.dtype)

                f.create_dataset('dF_F/average_dots_left', data=avg_df_F_left_dots, dtype=avg_df_F_left_dots.dtype)
                f.create_dataset('dF_F/average_dots_right', data=avg_df_F_right_dots, dtype=avg_df_F_right_dots.dtype)





            #smoothing

            smooth_avg_activity_left = savgol_filter(avg_df_F_left_dots, 20, 3)
            smooth_avg_activity_right = savgol_filter(avg_df_F_right_dots, 20, 3)
            smooth_trials_left = savgol_filter(df_F_left_dots, 20, 3, axis=1)
            smooth_trials_right = savgol_filter(df_F_right_dots, 20, 3, axis=1)

            #create folder for all dynamics
            os.makedirs(base_export_folder.joinpath('all_dynamics_average'),exist_ok=True)
            os.makedirs(base_export_folder.joinpath('all_dynamics_average_and_single_trials'),exist_ok=True)

            # Define time axis in seconds

            time_axis = np.arange(len(avg_df_F_right_dots)) * dt

            # Plot smoothed average activity
            fig, ax = plt.subplots()
            plt.plot(time_axis, smooth_avg_activity_left, color='blue', alpha=0.7, linewidth=3, label='Smoothed Average Left')
            plt.plot(time_axis, smooth_avg_activity_right, color='red', alpha=0.7, linestyle='--', linewidth=3, label='Smoothed Average Right')
            # Plot individual trial data with thin black lines for left and dashed black lines for right

            # Overlay shaded rectangle for stimulus epoch
            plt.axvspan(10, 50, color='gray', alpha=0.1, label='Stimulus Epoch')
            plt.title(f'Average and Individual Trial Activity Dynamics\nNeuron {cell.cellname}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Activity')
            # Set font of legend text to Arial
            legend = plt.legend(frameon=False, loc='upper left', fontsize='small')
            for text in legend.get_texts():
                text.set_fontfamily('Arial')
            # Set aspect ratio to 1
            ratio = 1.0
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            plt.savefig(temp_cell_path.joinpath(cell.cellname+'_rdms_average.png'))
            plt.savefig(base_export_folder.joinpath('all_dynamics_average').joinpath(cell.cellname + '_rdms_average.png'))
            plt.show()
            # Plot smoothed average activity with thin lines
            fig, ax = plt.subplots()
            plt.plot(time_axis, smooth_avg_activity_left, color='blue', alpha=0.7, linewidth=3, label='Smoothed Average Left')
            plt.plot(time_axis, smooth_avg_activity_right, color='red', alpha=0.7, linestyle='--', linewidth=3, label='Smoothed Average Right')
            # Plot individual trial data with thin black lines for left and dashed black lines for right
            for trial_left, trial_right in zip(smooth_trials_left, smooth_trials_right):
                plt.plot(time_axis, trial_left, color='black', alpha=0.3, linewidth=1)
                plt.plot(time_axis, trial_right, color='black', alpha=0.3, linestyle='--', linewidth=1)
            # Overlay shaded rectangle for stimulus epoch
            plt.axvspan(10, 50, color='gray', alpha=0.1, label='Stimulus Epoch')
            plt.title(f'Average and Individual Trial Activity Dynamics\nNeuron {cell.cellname}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Activity')
            # Set font of legend text to Arial
            legend = plt.legend(frameon=False, loc='upper left', fontsize='small')
            for text in legend.get_texts():
                text.set_fontfamily('Arial')
            # Set aspect ratio to 1
            ratio = 1.0
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            plt.savefig(temp_cell_path.joinpath(cell.cellname + '_rdms_average_and_single_trials.png'))
            plt.savefig(base_export_folder.joinpath('all_dynamics_average_and_single_trials').joinpath(cell.cellname + '_rdms_average_and_single_trials.png'))


        plt.show()
