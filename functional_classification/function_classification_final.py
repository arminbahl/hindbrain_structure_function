import h5py
import numpy as np
from pathlib import Path
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *

import pandas as pd
import pylab as pl
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
import re
import navis
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from scipy import stats
import seaborn as sns
from sklearn.cluster import KMeans
import scipy
from matplotlib.patches import Patch


def cronbach_alpha(data):
    # Number of trials
    N = data.shape[0]

    # Variance across time points (axis 0 means we calculate variance for each time point across all trials)
    variances_time_points = np.var(data, axis=0, ddof=1)

    # Variance of the total (flatten the array to consider all values)
    variance_total = np.var(data, ddof=1)

    # Cronbach's alpha calculation
    alpha = (N / (N - 1)) * (1 - np.sum(variances_time_points) / variance_total)

    return alpha
#classifying the functional dynamics using regressors and kmeans 2 wrrite to metadata
if __name__ == "__main__":
    # set variables
    np.set_printoptions(suppress=True)
    width_brain = 495.56
    data_path = Path(('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data'))
    data_path = Path(r'D:\hindbrain_structure_function\nextcloud')

    regressors = np.load(data_path / 'paGFP' /  "regressors_old.npy")
    regressors = regressors[:,:120]
    np.savetxt(data_path / f"regressors_old.txt", regressors, delimiter='\t')

    dt = 0.5

    #load all cell infortmation
    cell_data = load_cells_predictor_pipeline(path_to_data=Path(data_path), modalities=['clem','pa'], load_repaired=True)
    cell_data = cell_data.drop_duplicates(subset='cell_name')
    cell_data = cell_data.loc[cell_data['function'].isin(['integrator', 'dynamic_threshold', 'motor_command', 'dynamic threshold','motor command'])]

    custom_cutoff = {'integrator':0.85, 'dynamic_threshold':0.75, 'motor_command':0.85}
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

                df_F_left_single_trial= np.array(f['dF_F/single_trial_dots_left'])
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

            #time constant
            peak = 0.90 * np.nanmax(PD[20:100])  # 90% of the peak
            peak_indices = np.where(PD[20:100] >= peak)[0]

            rel = np.nanmean(np.mean(st,axis=0)/np.nanstd(st,axis=0), axis=0)
            ca = cronbach_alpha(st)

            # Compute the correleation coefficient to all three regressors
            ci = [np.corrcoef(PD[20:-20], regressors[0][20:-20])[0, 1],
                  np.corrcoef(PD[20:-20], regressors[1][20:-20])[0, 1],
                  np.corrcoef(PD[20:-20], regressors[2][20:-20])[0, 1]]
            print(directory,ci,'\n')
            st_best_fit = 0

            for i in range(st.shape[0]):
                if len(st.shape) == 1:
                    ci_st = [np.corrcoef(st[20:-20], regressors[0][20:-20])[0, 1],
                             np.corrcoef(st[20:-20], regressors[1][20:-20])[0, 1],
                             np.corrcoef(st[20:-20], regressors[2][20:-20])[0, 1]]
                else:
                    ci_st = [np.corrcoef(st[i, 20:-20], regressors[0][20:-20])[0, 1],
                             np.corrcoef(st[i, 20:-20], regressors[1][20:-20])[0, 1],
                             np.corrcoef(st[i, 20:-20], regressors[2][20:-20])[0, 1]]

                if max(ci_st)>st_best_fit:
                    st_best_fit = max(ci_st)
                    st_class_label = ['integrator', 'dynamic_threshold', 'motor_command'][np.argmax(ci_st)]





            class_label = ['integrator','dynamic_threshold','motor_command'][np.argmax(ci)]




            meta = open(data_path / 'paGFP' / directory / f'{directory}metadata.txt', 'r')
            t = meta.read()
            manual_class = eval(t.split("\n")[1][19:])[0].replace(" ", "_")
            if not t[-1:] == '\n':
                t = t + '\n'

            new_t = (t)
            meta.close()

            meta = open(data_path / 'paGFP' / directory / f'{directory}_metadata_with_regressor.txt', 'w')
            meta.write(new_t)
            meta.close()



            print(directory)
            temp_df = pd.DataFrame({'cell_name':[directory],
                                    'reliability':rel,
                                    'time_constant':peak_indices[0]})

            for it in ['PD', 'ND']:
                temp_df[it] = None
                temp_df[it] = temp_df[it].astype(object)

            temp_df['PD'] = [PD]
            temp_df["ND"] = [ND]




            if df is None:
                df = temp_df
            else:
                df = pd.concat([df, temp_df])
            manual_assigned = eval(new_t.split('\n')[1][19:])[0]
            plt.plot(PD / np.nanmax(PD), label='avg')
            plt.plot(st.T / np.nanmax(st.T), alpha=0.2, color='gray', label='single trials')
            temp_regressor = regressors[np.argmax(ci)]
            temp_regressor = temp_regressor + (0 - temp_regressor[0])
            plt.plot(temp_regressor / np.max(temp_regressor), label='best fit regressor')
            plt.legend(fontsize='xx-small', frameon=False)
            plt.title(f'{directory}\nManual: {manual_assigned}\n Predicted: {class_label}')

            fig = plt.gcf()
            os.makedirs(data_path / 'make_figures_FK_output' / "regressors_on_paGFP",exist_ok=True)
            fig.savefig(data_path / 'make_figures_FK_output' / "regressors_on_paGFP" / (directory + '.png'))
            plt.clf()
            fig.clf()
    try:
        df = df.reset_index(drop=True)
    except:
        pass


    #CLEM cells

    cells = os.listdir(data_path / 'clem_zfish1' / 'functionally_imaged')
    base_path_clem = data_path / 'clem_zfish1' / 'functionally_imaged'
    clem_rel = h5py.File(data_path / r"clem_zfish1/activity_recordings/all_cells_temp.h5")
    cells = [x for x in cells if (base_path_clem /x/ (f'{x}_dynamics.hdf5')).exists()]

    for directory in cells:
        if directory[12:] == 'cell_576460752652865344':
            pass
        if directory[12:] in list(cell_data.cell_name):
            with open((base_path_clem / directory / f"{directory}_metadata.txt" ), 'r') as f:
                t = f.read()
                neuron_functional_id = t.split('\n')[6].split(' ')[2].strip('"')
                neuron_functional_id = f'neuron_{neuron_functional_id}'
                manual_class = eval(t.split("\n")[7][19:])[0].replace(" ", "_")


            swc = navis.read_swc(data_path / 'clem_zfish1'/ 'functionally_imaged' / directory /'mapped'/ f'{directory}_mapped.swc')
            left_hemisphere = swc.nodes.iloc[0]['x'] < width_brain / 2
            temp_path = data_path / 'clem_zfish1'/ 'functionally_imaged' / directory / f'{directory}_dynamics.hdf5'
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


            #time constant
            # rel = np.array(clem_rel[neuron_functional_id]['reliability_trials_left'])[np.newaxis][0]
            rel = np.nanmean(np.mean(st,axis=0)/np.nanstd(st,axis=0), axis=0)
            peak = 0.90 * np.nanmax(PD[40:120])  # 90% of the peak
            if np.nanmax(PD[40:120])<0:
                peak = 1.1 * np.nanmax(PD[40:120])

            peak_indices = np.where(PD[40:120] >= peak)[0]

            ca = cronbach_alpha(st)

            # Compute the correleation coefficient to all three regressors
            ci = [np.corrcoef(PD[40:-40], regressors[0][20:-20])[0, 1],
                  np.corrcoef(PD[40:-40], regressors[1][20:-20])[0, 1],
                  np.corrcoef(PD[40:-40], regressors[2][20:-20])[0, 1]]
            print(directory, ci, '\n')
            st_best_fit = 0

            for i in range(st.shape[0]):

                if len(st.shape) == 1:
                    ci_st = [np.corrcoef(st[20:-20], regressors[0][20:-20])[0, 1],
                             np.corrcoef(st[20:-20], regressors[1][20:-20])[0, 1],
                             np.corrcoef(st[20:-20], regressors[2][20:-20])[0, 1]]
                else:

                    ci_st = [np.corrcoef(st[i,20:-20], regressors[0][20:-20])[0, 1],
                          np.corrcoef(st[i,20:-20], regressors[1][20:-20])[0, 1],
                          np.corrcoef(st[i,20:-20], regressors[2][20:-20])[0, 1]]

                if max(ci_st) > st_best_fit:
                    st_best_fit = max(ci_st)
                    st_class_label = ['integrator', 'dynamic_threshold', 'motor_command'][np.argmax(ci_st)]


            class_label = ['integrator', 'dynamic_threshold', 'motor_command'][np.argmax(ci)]

            meta = open(data_path / 'clem_zfish1'/ 'functionally_imaged' / directory / f'{directory}_metadata.txt', 'r')
            t = meta.read()
            if not t[-1:] == '\n':
                t = t + '\n'

            new_t = (t)
            meta.close()
            if (data_path / 'clem_zfish1'/ 'functionally_imaged' / directory).exists():
                meta = open(data_path / 'clem_zfish1'/ 'functionally_imaged' / directory / f'{directory}_metadata_with_regressor.txt', 'w')
                meta.write(new_t)
                meta.close()
            if (data_path / 'clem_zfish1' / 'functionally_imaged' / directory).exists():
                meta = open(data_path / 'clem_zfish1' / 'functionally_imaged' / directory / f'{directory}_metadata_with_regressor.txt', 'w')
                meta.write(new_t)
                meta.close()



            st_class_label
            temp_df = pd.DataFrame({'cell_name': [directory],
                                    'reliability':rel,
                                    'time_constant':peak_indices[0]})

            for it in ['PD', 'ND']:
                temp_df[it] = None
                temp_df[it] = temp_df[it].astype(object)

            temp_df['PD'] = [PD[20:-20]]
            temp_df["ND"] = [ND[20:-20]]

            if df is None:
                df = temp_df
            else:
                df = pd.concat([df, temp_df])
            manual_assigned = eval(new_t.split('\n')[7][19:])[0]
            plt.plot(PD[20:-20] / np.nanmax(PD[20:-20]), label='avg')
            plt.plot(st.T / np.nanmax(st.T), alpha=0.2, color='gray', label='single trials')
            temp_regressor = regressors[np.argmax(ci)]
            temp_regressor = temp_regressor + (0 - temp_regressor[0])
            plt.plot(temp_regressor / np.max(temp_regressor), label='best fit regressor')
            plt.legend(fontsize='xx-small', frameon=False)
            plt.title(f'{directory}\nManual: {manual_assigned}\n Predicted: {class_label}')


            fig = plt.gcf()
            os.makedirs(data_path / 'make_figures_FK_output' / "regressors_on_CLEM", exist_ok=True)
            fig.savefig(data_path / 'make_figures_FK_output' / "regressors_on_CLEM" / (directory + '.png'))
            plt.clf()
            fig.clf()
    try:
        df = df.reset_index(drop=True)
    except:
        pass










    all_PD = np.stack(df.PD.to_numpy())

    for i in range(all_PD.shape[0]):
        for i2 in range(all_PD[i].shape[0]):
            all_PD[i][::-1]
            if np.isnan(all_PD[i][::-1][i2]):
                all_PD[i][::-1][i2] = all_PD[i][::-1][i2-1]
    all_PD = (all_PD - np.nanmin(all_PD, axis=1)[:, np.newaxis]) / (np.nanmax(all_PD, axis=1)[:, np.newaxis] - np.nanmin(all_PD, axis=1)[:, np.newaxis])




    #Kmeans clustering
    n_clusters =3
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(all_PD)

    label2class = {'01':'motor_command','1n':'dynamic_threshold','2n':'integrator','00':'integrator'}
    int2class = {0:'integrator',1:'motor_command',2:'dynamic_threshold',3:'integrator'}
    df['kmeans_labels_int_1st'] = kmeans.labels_
    n_clusters_2nd =2
    kmeans_2nd = KMeans(n_clusters=n_clusters_2nd, random_state=0)
    kmeans_2nd.fit(all_PD[df['kmeans_labels_int_1st'] == 0])
    df['kmeans_labels_int_2nd'] = 'n'
    df.loc[df['kmeans_labels_int_1st']==0,'kmeans_labels_int_2nd'] = kmeans_2nd.labels_
    df['kmeans_labels_int_1st'] = df['kmeans_labels_int_1st'].astype(str)
    df['kmeans_labels_int_2nd'] = df['kmeans_labels_int_2nd'].astype(str)
    df['kmeans_labels_final'] = df['kmeans_labels_int_1st'] + df['kmeans_labels_int_2nd']
    for i,lf in enumerate(np.unique(df['kmeans_labels_final'])):
        df.loc[df['kmeans_labels_final']==lf,'kmeans_labels_int'] = int(i)


    fig, ax = plt.subplots(4, 1, figsize=(5, 3.5 * 4))
    for i,final_classes in enumerate(np.unique(df['kmeans_labels_final'])):
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

    kk = df.loc[:,['cell_name','kmeans_labels_final','kmeans_labels_int']]
    kk.columns = ['cell_name', 'kmeans_labels', 'kmeans_labels_int']
    kk['cell_name'] = kk['cell_name'].apply(lambda x: x[12:] if 'cell' in x else x)


    em_pa_cells = load_cells_predictor_pipeline(path_to_data=Path(data_path), modalities=['clem', 'pa'], load_repaired=True)
    em_pa_cells = em_pa_cells.loc[em_pa_cells['function']!='neg_control',:]


    for i in range(np.max(int(np.max(kk['kmeans_labels_int'])+1))):
        em_pa_cells.loc[em_pa_cells['cell_name'].isin(kk.loc[kk['kmeans_labels_int']==i,'cell_name']),'kmeans_functional_label'] = int(i)
        em_pa_cells.loc[em_pa_cells['cell_name'].isin(kk.loc[kk['kmeans_labels_int'] == i, 'cell_name']), 'kmeans_functional_label_str'] = int2class[i]


    accepted_function = ['integrator', 'motor_command', 'dynamic_threshold', 'integrator','dynamic threshold','motor command']

    for i,cell in em_pa_cells.iterrows():
        if cell['kmeans_functional_label_str'] is not np.nan and cell['kmeans_functional_label_str']!= 'nan' and cell.function in accepted_function:

            meta_path = Path(str(cell.metadata_path)[:-4] + '_with_regressor.txt')
            if not meta_path.parent.exists() or str(meta_path)=='.' or str(meta_path)=='_with_regressor.txt':
                meta_path = (data_path / 'paGFP' / cell['cell_name'] / f'{cell["cell_name"]}_metadata_with_regressor.txt')
            with open(meta_path, 'r') as meta:
                t = meta.read()
                if not t[-1:] == '\n':
                    t = t + '\n'

                prediction_string = f"kmeans_predicted_class = {cell['kmeans_functional_label_str']}\n"

            new_t = (t + prediction_string)

            if (data_path / 'clem_zfish1' / 'functionally_imaged'/f'clem_zfish1_{cell.cell_name}').exists():
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
    #figure showing reliability per kmeans cluster with modality split scatter
    plt.figure(dpi=300)
    in_legend = []
    np.random.seed(1)
    for i,item in df.iterrows():
        offset=-0.2+np.random.choice(np.arange(-0.075,0.075,0.01))
        marker = 'o'
        if item.imaging_modality == 'photoactivation':
            offset = 0.2+np.random.choice(np.arange(-0.075,0.075,0.01))
            marker = 's'

        if int2class[item.kmeans_labels_int] + " " + item.imaging_modality not in in_legend:
            plt.scatter(item.kmeans_labels_int+offset,item.reliability,c=color_dict[int2class[item.kmeans_labels_int]],label = int2class[item.kmeans_labels_int] + " " + item.imaging_modality,marker=marker)
            in_legend.append(int2class[item.kmeans_labels_int] + " " + item.imaging_modality)
        else:
            plt.scatter(item.kmeans_labels_int + offset, item.reliability, c=color_dict[int2class[item.kmeans_labels_int]],marker=marker)
        legend = plt.legend(frameon=False,fontsize='xx-small')
    plt.ylim(-2,7)
    plt.show()

    #figure showing reliability per kmeans cluster with modality split barplot
    plt.figure(dpi=300)
    in_legend = []
    x_ticks_labels = []
    loc = 0
    for kli in df.kmeans_labels_int.unique():
        for im in df.imaging_modality.unique():

            loc+=1
            plt.bar(loc,np.mean(df.loc[(df['kmeans_labels_int']==kli)&(df['imaging_modality']==im),'reliability']),
                      yerr=scipy.stats.sem(df.loc[(df['kmeans_labels_int']==kli)&(df['imaging_modality']==im),'reliability']),
                      edgecolor=color_dict[int2class[kli]],
                      color='white')
            x_ticks_labels.append(im[0] +"_"+int2class[kli][0])

            for i, item in df.loc[(df['kmeans_labels_int']==kli)&(df['imaging_modality']==im),:].iterrows():
                scatter_loc = loc + np.random.choice(np.arange(-0.075,0.075,0.01))
                marker = 'o'
                if item.imaging_modality == 'photoactivation':
                    offset = 0.2 + np.random.choice(np.arange(-0.075, 0.075, 0.01))
                    marker = 's'


                plt.scatter(scatter_loc, item.reliability, c=color_dict[int2class[item.kmeans_labels_int]], marker=marker)

    from matplotlib.lines import Line2D
    in_legend = [Line2D([0], [0], marker='o', color='k', label='CLEM',markerfacecolor='w', linestyle='None'),
                 Line2D([0], [0], marker='s', color='k', label='Photoactivation',markerfacecolor='w', linestyle='None'),
                 Patch(facecolor='#64c5ebb3', edgecolor='#64c5ebb3',label='Dynamic Threshold'),
                 Patch(facecolor='#e84d8ab3', edgecolor='#e84d8ab3', label='Integrator'),
                 Patch(facecolor='#7f58afb3', edgecolor='#7f58afb3', label='Motor Command')]
    plt.legend(handles=in_legend,frameon=False, fontsize='x-small')
    plt.ylim(-2, 7)
    plt.show()





    #save class assignment
    os.makedirs(data_path / 'make_figures_FK_output' / 'functional_analysis', exist_ok=True)
    em_pa_cells.loc[:,['cell_name','functional_id','kmeans_functional_label_str']].to_excel(data_path / 'make_figures_FK_output' / 'functional_analysis'/'assignment_.xlsx')


    #save regressor
    regressors = np.vstack([kmeans.cluster_centers_[1:],kmeans_2nd.cluster_centers_])
    np.save(data_path / 'make_figures_FK_output' / 'functional_analysis'/ 'kmeans_regressors.npy',regressors,)
