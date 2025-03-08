import re
import h5py
import matplotlib.pyplot as plt
import scipy
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.cluster import KMeans
import sys
import os
sys.path.extend(['/Users/fkampf/PycharmProjects'])
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
import chardet



# classifying the functional dynamics using regressors and kmeans 2 write to metadata
def get_encoding(path):
    with open(path, 'rb') as f:
        t = f.read()
        r = chardet.detect(t)
        return r['encoding']


if __name__ == "__main__":
    # set variables

    write_metadata = True
    np.set_printoptions(suppress=True)
    width_brain = 495.56


    data_path = Path(('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data'))
    data_path = Path(r'D:\hindbrain_structure_function\nextcloud')
    data_path = Path('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud')
    savepath = data_path / 'make_figures_FK_output' / 'functional_analysis'
    os.makedirs(savepath, exist_ok=True)


    # load all cell infortmation
    cell_data = load_cells_predictor_pipeline(path_to_data=Path(data_path), modalities=['clem', 'pa'], load_repaired=True)
    cell_data = cell_data.drop_duplicates(subset='cell_name')
    cell_data = cell_data.loc[cell_data['function'].isin(['integrator', 'dynamic_threshold', 'motor_command', 'dynamic threshold', 'motor command'])]

    custom_cutoff = {'integrator': 0.85, 'dynamic_threshold': 0.75, 'motor_command': 0.85}
    # Define the pattern
    pattern = re.compile(r'^\d{8}\.\d$')

    cells = os.listdir(data_path / 'paGFP')

    # Filter files based on the pattern
    cells = [f for f in cells if pattern.match(f)]
    df = None
    for directory in cells:
        if directory in list(cell_data.cell_name):
            swc = navis.read_swc(data_path / 'paGFP' / directory / f'{directory}.swc')
            left_hemisphere = swc.nodes.iloc[0]['x'] < width_brain / 2
            temp_path = data_path / 'paGFP' / directory / f'{directory}_dynamics.hdf5'
            with h5py.File(temp_path, 'r') as f:
                df_F_left_dots_avg = np.array(f['dF_F/average_dots_left'])
                df_F_right_dots_avg = np.array(f['dF_F/average_dots_right'])

                df_F_left_single_trial = np.array(f['dF_F/single_trial_dots_left'])
                df_F_right_single_trial = np.array(f['dF_F/single_trial_dots_right'])

            # As we have the cell registered to the z-brain, we know if it is on the left or right hemisphere
            if left_hemisphere:
                PD = df_F_left_dots_avg  # We drop the first and last 10 s, as this is how the regressors had been computed
                ND = df_F_right_dots_avg
                st = df_F_left_single_trial
            else:
                PD = df_F_right_dots_avg
                ND = df_F_left_dots_avg
                st = df_F_right_single_trial

            # time constant
            peak = 0.90 * np.nanmax(PD[20:100])  # 90% of the peak
            peak_indices = np.where(PD[20:100] >= peak)[0]
            direction_selectivity = np.round((max(PD[20:100]) - max(ND[20:100])) / (max(PD[20:100]) + max(ND[20:100])), 2)

            rel = np.round(np.nanmean(np.mean(st, axis=0) / np.nanstd(st, axis=0), axis=0),2)

            temp_df = pd.DataFrame([{'cell_name': directory,
                                    'reliability': rel,
                                    'time_constant': peak_indices[0],
                                     'direction_selectivity': direction_selectivity}])

            for it in ['PD', 'ND']:
                temp_df[it] = None
                temp_df[it] = temp_df[it].astype(object)

            temp_df['PD'] = [PD]
            temp_df["ND"] = [ND]

            if df is None:
                df = temp_df
            else:
                df = pd.concat([df, temp_df])

    try:
        df = df.reset_index(drop=True)
    except:
        pass

    # CLEM cells

    cells = os.listdir(data_path / 'clem_zfish1' / 'functionally_imaged')
    base_path_clem = data_path / 'clem_zfish1' / 'functionally_imaged'
    clem_rel = h5py.File(data_path / r"clem_zfish1/activity_recordings/all_cells_temp.h5")
    cells = [x for x in cells if (base_path_clem / x / (f'{x}_dynamics.hdf5')).exists()]

    for directory in cells:
        if directory[12:] in list(cell_data.cell_name):
            with open((base_path_clem / directory / f"{directory}_metadata.txt"), 'r') as f:
                t = f.read()
                neuron_functional_id = t.split('\n')[6].split(' ')[2].strip('"')
                neuron_functional_id = f'neuron_{neuron_functional_id}'

            swc = navis.read_swc(data_path / 'clem_zfish1' / 'functionally_imaged' / directory / 'mapped' / f'{directory}_mapped.swc')
            left_hemisphere = swc.nodes.iloc[0]['x'] < width_brain / 2
            temp_path = data_path / 'clem_zfish1' / 'functionally_imaged' / directory / f'{directory}_dynamics.hdf5'
            with h5py.File(temp_path, 'r') as f:
                df_F_left_dots_avg = np.array(f['dF_F/average_rdms_left_dF_F_calculated_on_single_trials'])
                df_F_right_dots_avg = np.array(f['dF_F/average_rdms_right_dF_F_calculated_on_single_trials'])
            df_F_left_single_trial = clem_rel[neuron_functional_id]['neuronal_activity_trials_left'][:, 20:140]
            df_F_right_single_trial = clem_rel[neuron_functional_id]['neuronal_activity_trials_right'][:, 20:140]

            # As we have the cell registered to the z-brain, we know if it is on the left or right hemisphere
            if left_hemisphere:
                PD = df_F_left_dots_avg  # We drop the first and last 10 s, as this is how the regressors had been computed
                ND = df_F_right_dots_avg
                st = df_F_left_single_trial
            else:
                PD = df_F_right_dots_avg
                ND = df_F_left_dots_avg
                st = df_F_right_single_trial

            # time constant
            rel = np.round(np.nanmean(np.mean(st, axis=0) / np.nanstd(st, axis=0), axis=0),2)
            peak = 0.90 * np.nanmax(PD[40:120])  # 90% of the peak
            if np.nanmax(PD[40:120]) < 0:
                peak = 1.1 * np.nanmax(PD[40:120])

            peak_indices = np.where(PD[40:120] >= peak)[0]
            direction_selectivity = np.round((max(PD[20:100]) - max(ND[20:100])) / (max(PD[20:100]) + max(ND[20:100])), 2)

            temp_df = pd.DataFrame([{'cell_name': "_".join(directory.split('_')[2:]),
                                    'reliability': rel,
                                    'time_constant': peak_indices[0],
                                     'direction_selectivity': direction_selectivity}])

            for it in ['PD', 'ND']:
                temp_df[it] = None
                temp_df[it] = temp_df[it].astype(object)

            temp_df['PD'] = [PD[20:-20]]
            temp_df["ND"] = [ND[20:-20]]

            if df is None:
                df = temp_df
            else:
                df = pd.concat([df, temp_df])
    try:
        df = df.reset_index(drop=True)
    except:
        pass

    all_PD = np.stack(df.PD.to_numpy())

    for i in range(all_PD.shape[0]):
        for i2 in range(all_PD[i].shape[0]):
            all_PD[i][::-1]
            if np.isnan(all_PD[i][::-1][i2]):
                all_PD[i][::-1][i2] = all_PD[i][::-1][i2 - 1]
    all_PD = (all_PD - np.nanmin(all_PD, axis=1)[:, np.newaxis]) / (np.nanmax(all_PD, axis=1)[:, np.newaxis] - np.nanmin(all_PD, axis=1)[:, np.newaxis])

    # Kmeans clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(all_PD)

    label2class = {'1n': 'dynamic_threshold', '2n': 'integrator', '01': 'integrator', '00': 'motor_command'}
    int2class = {2: 'dynamic_threshold', 1: 'integrator', 3: 'integrator', 0: 'motor_command'}
    df['kmeans_labels_int_1st'] = kmeans.labels_
    n_clusters_2nd = 2
    kmeans_2nd = KMeans(n_clusters=n_clusters_2nd, random_state=0)
    kmeans_2nd.fit(all_PD[df['kmeans_labels_int_1st'] == 0])
    df['kmeans_labels_int_2nd'] = 'n'
    df.loc[df['kmeans_labels_int_1st'] == 0, 'kmeans_labels_int_2nd'] = kmeans_2nd.labels_
    df['kmeans_labels_int_1st'] = df['kmeans_labels_int_1st'].astype(str)
    df['kmeans_labels_int_2nd'] = df['kmeans_labels_int_2nd'].astype(str)
    df['kmeans_labels_final'] = df['kmeans_labels_int_1st'] + df['kmeans_labels_int_2nd']
    for i, lf in enumerate(np.unique(df['kmeans_labels_final'])):
        df.loc[df['kmeans_labels_final'] == lf, 'kmeans_labels_int'] = int(i)




    #load new neurotransmitter
    new_neurotransmitter = pd.read_excel(
        data_path / 'em_zfish1' / 'Figures' / 'Fig 4' / 'cells2show.xlsx',
        sheet_name='paGFP stack quality', dtype=str)






    fig, ax = plt.subplots(4, 1, figsize=(5, 3.5 * 4))
    for i, final_classes in enumerate(np.unique(df['kmeans_labels_final'])):
        temp_bool = df['kmeans_labels_final'] == final_classes
        ax[i].title.set_text(f'cluster {final_classes}')
        ax[i].plot(all_PD[temp_bool, :].T)
    plt.show()

    df['kmeans_labels'] = [label2class[x] for x in df['kmeans_labels_final']]
    df['functional_id'] = np.nan
    df['imaging_modality'] = np.nan
    for i, cell in df.iterrows():
        try:
            functional_id_target = cell_data.loc[cell_data['cell_name'] == cell['cell_name'], 'functional_id'].iloc[0]
            imaging_modality = cell_data.loc[cell_data['cell_name'] == cell['cell_name'], 'imaging_modality'].iloc[0]
        except:
            functional_id_target = cell_data.loc[cell_data['cell_name'] == cell['cell_name'], 'functional_id'].iloc[0]
            imaging_modality = cell_data.loc[cell_data['cell_name'] == cell['cell_name'], 'imaging_modality'].iloc[0]
        df.loc[i, 'functional_id'] = functional_id_target
        df.loc[i, 'imaging_modality'] = imaging_modality

    neurotransmitter_dict = {'Vglut2a': 'excitatory', 'Gad1b': 'inhibitory'}
    for i, cell in df.iterrows():
        if cell.imaging_modality == 'photoactivation':
            if new_neurotransmitter.loc[new_neurotransmitter['Name'] == cell.cell_name, 'Neurotransmitter'].iloc[0] is np.nan:
                df.loc[i, 'neurotransmitter'] = 'nan'
            else:
                df.loc[i, 'neurotransmitter'] = neurotransmitter_dict[new_neurotransmitter.loc[
                    new_neurotransmitter['Name'] == cell.cell_name, 'Neurotransmitter'].iloc[0]]

    kk = df.loc[:, ['cell_name', 'kmeans_labels_final', 'kmeans_labels_int']]
    kk.columns = ['cell_name', 'kmeans_labels', 'kmeans_labels_int']


    em_pa_cells = load_cells_predictor_pipeline(path_to_data=Path(data_path), modalities=['clem', 'pa'], load_repaired=True)
    em_pa_cells = em_pa_cells.drop_duplicates(subset='cell_name')
    em_pa_cells = em_pa_cells.loc[em_pa_cells['function'].isin(['integrator', 'dynamic_threshold', 'motor_command', 'dynamic threshold', 'motor command'])]
    em_pa_cells = em_pa_cells.loc[em_pa_cells['function'] != 'neg_control', :]

    for i,cell in em_pa_cells.iterrows():
        em_pa_cells.loc[i, 'kmeans_functional_label'] = int(kk.loc[kk['cell_name']==cell.cell_name,'kmeans_labels_int'])
        em_pa_cells.loc[i, 'kmeans_functional_label_str'] = label2class[kk.loc[kk['cell_name']==cell.cell_name,'kmeans_labels'].iloc[0]]
        em_pa_cells.loc[i, 'kmeans_functional_label_subclass'] = kk.loc[kk['cell_name'] == cell.cell_name, 'kmeans_labels'].iloc[0]

    accepted_function = ['integrator', 'motor_command', 'dynamic_threshold', 'integrator', 'dynamic threshold', 'motor command']

    for i, cell in em_pa_cells.iterrows():
        if cell['kmeans_functional_label_str'] is not np.nan and cell['kmeans_functional_label_str'] != 'nan' and cell.function in accepted_function:

            meta_path = Path(str(cell.metadata_path)[:-4] + '.txt')
            if not meta_path.parent.exists() or str(meta_path) == '.' or str(meta_path) == '.txt':
                meta_path = (data_path / 'paGFP' / cell['cell_name'] / f'{cell["cell_name"]}metadata.txt')

            with open(str(meta_path), 'r', encoding=get_encoding(meta_path)) as meta:
                t = meta.read()
            if not t[-1:] == '\n':
                    t = t + '\n'

            prediction_string = f"kmeans_predicted_class = {cell['kmeans_functional_label_str']}\n"
            try:
                reliability_string = f"reliability = {df.loc[df['cell_name']==cell['cell_name'],'reliability'].values[0]}\n"
                direction_selectivity_string = f"direction_selectivity = {df.loc[df['cell_name'] == cell['cell_name'], 'direction_selectivity'].values[0]}\n"
                time_constant_string = f"time_constant = {df.loc[df['cell_name'] == cell['cell_name'], 'time_constant'].values[0]}\n"
            except:
                reliability_string = f"reliability = {df.loc[df['cell_name'] == 'clem_zfish1_' + cell['cell_name'], 'reliability'].values[0]}\n"
                direction_selectivity_string = f"direction_selectivity = {df.loc[df['cell_name'] == 'clem_zfish1_' + cell['cell_name'], 'direction_selectivity'].values[0]}\n"
                time_constant_string = f"time_constant = {df.loc[df['cell_name'] == 'clem_zfish1_' + cell['cell_name'], 'time_constant'].values[0]}\n"


            new_t = (t + prediction_string + reliability_string + direction_selectivity_string+time_constant_string)

            if (data_path / 'clem_zfish1' / 'functionally_imaged' / f'clem_zfish1_{cell.cell_name}').exists():
                temp_path = data_path / 'clem_zfish1' / 'functionally_imaged' / f'clem_zfish1_{cell.cell_name}' / f'clem_zfish1_{cell["cell_name"]}_metadata_with_regressor.txt'
                if write_metadata:
                    with open(temp_path, 'w') as meta:
                        meta.write(new_t)
            if (data_path / 'clem_zfish1' / 'all_cells' / f'clem_zfish1_{cell.cell_name}').exists():
                temp_path = data_path / 'clem_zfish1' / 'all_cells' / f'clem_zfish1_{cell.cell_name}' / f'clem_zfish1_{cell["cell_name"]}_metadata_with_regressor.txt'
                if write_metadata:
                    with open(temp_path, 'w') as meta:
                        meta.write(new_t)
            if (data_path / 'paGFP' / cell.cell_name).exists():
                temp_path = data_path / 'paGFP' / cell.cell_name / f'{cell["cell_name"]}_metadata_with_regressor.txt'
                if write_metadata:
                    with open(temp_path, 'w') as meta:
                        meta.write(new_t)

    color_dict = {
        "integrator": '#e84d8ab3',
        "dynamic_threshold": '#64c5ebb3',
        "motor_command": '#7f58afb3',
        'neg control': "#a8c256b3"
    }

    def plot_fig(df, color_dict: dict, int2class: dict, attribute: str,savepath, ylim=None):
        plt.figure(dpi=300)
        loc = 0
        for kli in df.kmeans_labels_int.unique():
            for im in df.imaging_modality.unique():
                loc += 1
                plt.bar(loc, np.mean(df.loc[(df['kmeans_labels_int'] == kli) & (df['imaging_modality'] == im), attribute]),
                        yerr=scipy.stats.sem(df.loc[(df['kmeans_labels_int'] == kli) & (df['imaging_modality'] == im), attribute]),
                        edgecolor=color_dict[int2class[kli]],
                        color='white')
                for i, item in df.loc[(df['kmeans_labels_int'] == kli) & (df['imaging_modality'] == im), :].iterrows():
                    scatter_loc = loc + np.random.choice(np.arange(-0.075, 0.075, 0.01))
                    marker = 's' if item.imaging_modality == 'photoactivation' else 'o'
                    plt.scatter(scatter_loc, item[attribute], c=color_dict[int2class[item.kmeans_labels_int]], marker=marker)
        in_legend = [Line2D([0], [0], marker='s', color='k', label='Photoactivation', markerfacecolor='w', linestyle='None'),
                     Line2D([0], [0], marker='o', color='k', label='CLEM', markerfacecolor='w', linestyle='None'),
                     Patch(facecolor='#64c5ebb3', edgecolor='#64c5ebb3', label='Dynamic Threshold'),
                     Patch(facecolor='#e84d8ab3', edgecolor='#e84d8ab3', label='Integrator'),
                     Patch(facecolor='#7f58afb3', edgecolor='#7f58afb3', label='Motor Command')]
        plt.legend(handles=in_legend, frameon=False, fontsize='x-small')
        if ylim is not None:
            plt.ylim(ylim)
        plt.title(attribute)
        plt.savefig(savepath/(attribute+'.png'))
        plt.savefig(savepath / (attribute + '.pdf'))
        plt.show()


    # Now you can call the function for each attribute
    plot_fig(df.sort_values('kmeans_labels_int'), color_dict, int2class, 'reliability', savepath=savepath, ylim=(-2, 7))
    plot_fig(df.sort_values('kmeans_labels_int'), color_dict, int2class, 'time_constant', savepath=savepath)
    plot_fig(df.sort_values('kmeans_labels_int'), color_dict, int2class, 'direction_selectivity', savepath=savepath)

    # save class assignment
    em_pa_cells.loc[:, ['cell_name', 'functional_id', 'kmeans_functional_label_str']].to_excel(savepath / 'assignment_.xlsx')

    # save regressor
    regressors = np.vstack([kmeans.cluster_centers_[1:], kmeans_2nd.cluster_centers_])
    np.save(savepath / 'kmeans_regressors.npy', regressors, )

    # add morphology
    df = pd.merge(df, em_pa_cells.loc[:, ["cell_name", "morphology"]], on='cell_name', how='left')


    # create jon activity plots

    def jon_activity_plot(activity_array, cell_type, dt=0.5, color='red', savepath=savepath):
        fig, ax = plt.subplots()
        activity_array = ((activity_array - np.nanmin(activity_array, axis=1)[:, np.newaxis]) / (
                np.nanmax(activity_array, axis=1)[:, np.newaxis] - np.nanmin(activity_array, axis=1)[:,
                                                                   np.newaxis])) * 100

        plt.axvspan(10, 50, color='gray', alpha=0.1)
        time_axis = np.arange(activity_array.shape[1]) * dt
        for i in range(activity_array.shape[0]):
            plt.plot(time_axis, activity_array[i, :], color=color, alpha=0.7, linestyle='-', linewidth=1)

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

        # # Save the figure
        filename = f"{cell_type}_activity-traces.pdf"
        savepath = savepath / 'activity_plots4fig'
        os.makedirs(savepath, exist_ok=True)
        output_path = savepath / filename
        fig.savefig(output_path, dpi=1200)
        print(f"Figure saved successfully at: {output_path}")
    
    def flo_activity_plot(activity_array, cell_type, dt=0.5, color='red', savepath=savepath):
        fig, ax = plt.subplots()
        activity_array = ((activity_array - np.nanmin(activity_array, axis=1)[:, np.newaxis]) / (
                np.nanmax(activity_array, axis=1)[:, np.newaxis] - np.nanmin(activity_array, axis=1)[:,
                                                                   np.newaxis])) * 100

        plt.axvline(10, color='k', alpha=0.9, linestyle='--', linewidth=1)
        time_axis = np.arange(activity_array.shape[1]) * dt
        for i in range(activity_array.shape[0]):
            plt.plot(time_axis, activity_array[i, :], color='gray', alpha=0.5, linestyle='-', linewidth=1)
        plt.plot(time_axis, np.nanmean(activity_array, axis=0), color=color, alpha=1, linestyle='-', linewidth=3)

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

        # # Save the figure
        filename = f"{cell_type}_activity-traces_fk.pdf"
        savepath = savepath / 'activity_plots4fig'
        os.makedirs(savepath, exist_ok=True)
        output_path = savepath / filename
        fig.savefig(output_path, dpi=1200)
        print(f"Figure saved successfully at: {output_path}")


        


    unpack_activity_traces = lambda x: np.array([y for y in x])
    activity_dt = unpack_activity_traces(df.loc[(df.imaging_modality == 'photoactivation') &
                                                (df.kmeans_labels == 'dynamic_threshold'), "PD"].to_numpy())
    activity_mc = unpack_activity_traces(df.loc[(df.imaging_modality == 'photoactivation') &
                                                (df.kmeans_labels == 'motor_command'), "PD"].to_numpy())
    activity_ii = unpack_activity_traces(df.loc[(df.morphology == 'ipsilateral') &
                                                (df.imaging_modality == 'photoactivation') &
                                                (df.kmeans_labels == 'integrator'), "PD"].to_numpy())
    activity_ci = unpack_activity_traces(df.loc[(df.morphology == 'contralateral') &
                                                (df.imaging_modality == 'photoactivation') &
                                                (df.kmeans_labels == 'integrator'), "PD"].to_numpy())

    jon_activity_plot(activity_dt, cell_type='dynamic_threshold', color='#64c5ebb3')
    jon_activity_plot(activity_mc, cell_type='motor_command', color='#7f58afb3')
    jon_activity_plot(activity_ii, cell_type='ipsilateral_integrator', color='#feb326b3')
    jon_activity_plot(activity_ci, cell_type='contralateral_integrator', color='#e84d8ab3')

    flo_activity_plot

    flo_activity_plot(activity_dt, cell_type='dynamic_threshold', color='#64c5ebb3')
    flo_activity_plot(activity_mc, cell_type='motor_command', color='#7f58afb3')
    flo_activity_plot(activity_ii, cell_type='ipsilateral_integrator', color='#feb326b3')
    flo_activity_plot(activity_ci, cell_type='contralateral_integrator', color='#e84d8ab3')
    # neurotransmitter
    df['cell_class'] = df['kmeans_labels']

    df['cell_class'] = df.apply(
        lambda x: x['cell_class'] if x.kmeans_labels != 'integrator' else x.kmeans_labels + "_" + x.morphology, axis=1)

    import pandas as pd
    import matplotlib.pyplot as plt

    # Group and reset the index to make the grouped columns regular columns
    stacked_barplot = df.groupby(['imaging_modality', 'cell_class', 'neurotransmitter']).size().reset_index(
        name='count')

    # Fill NaN neurotransmitter values with 'unknown'
    stacked_barplot['neurotransmitter'] = stacked_barplot['neurotransmitter'].fillna('unknown')

    # Pivot the data for the stacked barplot
    pivot_data = stacked_barplot.pivot_table(index='cell_class', columns='neurotransmitter', values='count',
                                             aggfunc='sum', fill_value=0)

    # Normalize the bars (convert counts to proportions)
    normalized_data = pivot_data.div(pivot_data.sum(axis=1), axis=0)

    # Plotting with spacing between bars
    ax = normalized_data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis', width=0.8)

    # Add space between bars by reducing the bar width and shifting the positions slightly
    for bar_group in ax.containers:
        for bar in bar_group:
            bar.set_x(bar.get_x() + 0.05)  # Add space between bars

    # Adding labels and title
    plt.title('Normalized Stacked Bar Plot: Neurotransmitters per Cell Class')
    plt.xlabel('Cell Class')
    plt.ylabel('Proportion')
    plt.legend(title='Neurotransmitter', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    neurotransmitter_path = savepath / 'neurotransmitter_plot'
    os.makedirs(neurotransmitter_path, exist_ok=True)
    plt.savefig(neurotransmitter_path / 'neurotransmitter_stacked_barplot.pdf')
    plt.show()
