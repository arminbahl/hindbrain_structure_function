import re
import h5py
import matplotlib.pyplot as plt
import scipy
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from sklearn.cluster import KMeans
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *

# classifying the functional dynamics using regressors and kmeans 2 wrrite to metadata
if __name__ == "__main__":
    # set variables
    np.set_printoptions(suppress=True)
    width_brain = 495.56


    data_path = Path(('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data'))
    data_path = Path(r'D:\hindbrain_structure_function\nextcloud')
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

            temp_df = pd.DataFrame({'cell_name': [directory],
                                    'reliability': rel,
                                    'time_constant': peak_indices[0],
                                    'direction_selectivity': direction_selectivity})

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

            temp_df = pd.DataFrame({'cell_name': [directory],
                                    'reliability': rel,
                                    'time_constant': peak_indices[0],
                                    'direction_selectivity': direction_selectivity})

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

    label2class = {'01': 'motor_command', '1n': 'dynamic_threshold', '2n': 'integrator', '00': 'integrator'}
    int2class = {0: 'integrator', 1: 'motor_command', 2: 'dynamic_threshold', 3: 'integrator'}
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
            functional_id_target = cell_data.loc[cell_data['cell_name'] == cell['cell_name'][12:], 'functional_id'].iloc[0]
            imaging_modality = cell_data.loc[cell_data['cell_name'] == cell['cell_name'][12:], 'imaging_modality'].iloc[0]
        df.loc[i, 'functional_id'] = functional_id_target
        df.loc[i, 'imaging_modality'] = imaging_modality

    kk = df.loc[:, ['cell_name', 'kmeans_labels_final', 'kmeans_labels_int']]
    kk.columns = ['cell_name', 'kmeans_labels', 'kmeans_labels_int']
    kk['cell_name'] = kk['cell_name'].apply(lambda x: x[12:] if 'cell' in x else x)

    em_pa_cells = load_cells_predictor_pipeline(path_to_data=Path(data_path), modalities=['clem', 'pa'], load_repaired=True)
    em_pa_cells = em_pa_cells.loc[em_pa_cells['function'] != 'neg_control', :]

    for i in range(np.max(int(np.max(kk['kmeans_labels_int']) + 1))):
        em_pa_cells.loc[em_pa_cells['cell_name'].isin(kk.loc[kk['kmeans_labels_int'] == i, 'cell_name']), 'kmeans_functional_label'] = int(i)
        em_pa_cells.loc[em_pa_cells['cell_name'].isin(kk.loc[kk['kmeans_labels_int'] == i, 'cell_name']), 'kmeans_functional_label_str'] = int2class[i]

    accepted_function = ['integrator', 'motor_command', 'dynamic_threshold', 'integrator', 'dynamic threshold', 'motor command']

    for i, cell in em_pa_cells.iterrows():
        if cell['kmeans_functional_label_str'] is not np.nan and cell['kmeans_functional_label_str'] != 'nan' and cell.function in accepted_function:

            meta_path = Path(str(cell.metadata_path)[:-4] + '.txt')
            if not meta_path.parent.exists() or str(meta_path) == '.' or str(meta_path) == '.txt':
                meta_path = (data_path / 'paGFP' / cell['cell_name'] / f'{cell["cell_name"]}metadata.txt')


            with open(str(meta_path), 'r') as meta:
                t = meta.read()
            if not t[-1:] == '\n':
                    t = t + '\n'

            prediction_string = f"kmeans_predicted_class = {cell['kmeans_functional_label_str']}\n"
            try:
                reliability_string = f"reliability = {df.loc[df['cell_name']==cell['cell_name'],'reliability'].values[0]}\n"
                direction_selectivity_string = f"direction_selectivity = {df.loc[df['cell_name'] == cell['cell_name'], 'direction_selectivity'].values[0]}\n"
            except:
                reliability_string = f"reliability = {df.loc[df['cell_name'] == 'clem_zfish1_' + cell['cell_name'], 'reliability'].values[0]}\n"
                direction_selectivity_string = f"direction_selectivity = {df.loc[df['cell_name'] == 'clem_zfish1_' + cell['cell_name'], 'direction_selectivity'].values[0]}\n"


            new_t = (t + prediction_string + reliability_string + direction_selectivity_string)

            if (data_path / 'clem_zfish1' / 'functionally_imaged' / f'clem_zfish1_{cell.cell_name}').exists():
                temp_path = data_path / 'clem_zfish1' / 'functionally_imaged' / f'clem_zfish1_{cell.cell_name}' / f'clem_zfish1_{cell["cell_name"]}_metadata_with_regressor.txt'
                with open(temp_path, 'w') as meta:
                    meta.write(new_t)
            if (data_path / 'clem_zfish1' / 'all_cells' / f'clem_zfish1_{cell.cell_name}').exists():
                temp_path = data_path / 'clem_zfish1' / 'all_cells' / f'clem_zfish1_{cell.cell_name}' / f'clem_zfish1_{cell["cell_name"]}_metadata_with_regressor.txt'
                with open(temp_path, 'w') as meta:
                    meta.write(new_t)
            if (data_path / 'paGFP' / cell.cell_name).exists():
                temp_path = data_path / 'paGFP' / cell.cell_name / f'{cell["cell_name"]}_metadata_with_regressor.txt'
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
    plot_fig(df, color_dict, int2class, 'reliability',savepath=savepath, ylim = (-2, 7))
    plot_fig(df, color_dict, int2class, 'time_constant',savepath=savepath)
    plot_fig(df, color_dict, int2class, 'direction_selectivity',savepath=savepath)

    # save class assignment
    em_pa_cells.loc[:, ['cell_name', 'functional_id', 'kmeans_functional_label_str']].to_excel(savepath / 'assignment_.xlsx')

    # save regressor
    regressors = np.vstack([kmeans.cluster_centers_[1:], kmeans_2nd.cluster_centers_])
    np.save(savepath / 'kmeans_regressors.npy', regressors, )