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

    regressors = np.load(data_path / 'paGFP' /  "regressors_old.npy")
    regressors = regressors[:,:120]
    np.savetxt(data_path / f"regressors_old.txt", regressors, delimiter='\t')

    dt = 0.5

    #load all cell infortmation
    cell_data = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem','pa'], load_repaired=True)
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


            prediction_string = f'regressor_predicted_class = "{class_label}"\n'
            correlation_test = f'correlation_test_passed = "{ci[np.argmax(ci)] > custom_cutoff[class_label]}"\n'
            prediction_string_single_trial = f'regressor_predicted_class_single_trial = "{st_class_label}"\n'
            correlation_test_single_trial = f'correlation_test_single_trial_passed = {ci[np.argmax(ci_st)] > custom_cutoff[st_class_label]}\n'



            meta = open(data_path / 'paGFP' / directory / f'{directory}metadata.txt', 'r')
            t = meta.read()
            manual_class = eval(t.split("\n")[1][19:])[0].replace(" ", "_")
            prediction_equals_manual = f'prediction_equals_manual = {manual_class == class_label}\n'
            prediction_equals_manual_st = f'prediction_equals_manual = {manual_class == st_class_label}\n'
            if not t[-1:] == '\n':
                t = t + '\n'

            new_t = (t + prediction_string + correlation_test + prediction_equals_manual + prediction_string_single_trial +correlation_test_single_trial+prediction_equals_manual_st)
            meta.close()

            meta = open(data_path / 'paGFP' / directory / f'{directory}_metadata_with_regressor.txt', 'w')
            meta.write(new_t)
            meta.close()



            temp_df = pd.DataFrame({'cell_name':[directory],
                                    'manual_assigned_class':[eval(new_t.split('\n')[1][19:])[0]],
                                    'predicted_class':[class_label],
                                    'predicted_class_single_trial': [st_class_label],
                                    'max_correlation':[ci[np.argmax(ci)]],
                                    'max_correlation_single_trial': st_best_fit,
                                    'correlation_mc': np.corrcoef(PD[20:-20], regressors[2][20:-20])[0, 1],
                                    'correlation_dt': np.corrcoef(PD[20:-20], regressors[1][20:-20])[0, 1],
                                    'correlation_i': np.corrcoef(PD[20:-20], regressors[0][20:-20])[0, 1],
                                    'passed_correlation':[ci[np.argmax(ci)] > custom_cutoff[class_label]],
                                    'passed_correlation_single_trial': [st_best_fit > custom_cutoff[st_class_label]],
                                    'manual_matches_regressor':[eval(new_t.split('\n')[1][19:])[0]==class_label],
                                    'manual_matches_regressor_St': [eval(new_t.split('\n')[1][19:])[0].replace(" ", "_") == st_class_label],
                                    'modality':'pa',
                                    'reliability':rel,
                                    'cronbach':ca,
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
    clem_rel = h5py.File(r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\clem_zfish1\activity_recordings\all_cells_temp.h5")
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

            prediction_string = f'regressor_predicted_class = "{class_label}"\n'
            correlation_test = f'correlation_test_passed = {ci[np.argmax(ci)] > custom_cutoff[class_label]}\n'
            prediction_string_single_trial = f'regressor_predicted_class_single_trial = "{class_label}"\n'
            correlation_test_single_trial = f'correlation_test_single_trial_passed = {ci[np.argmax(ci)] > custom_cutoff[st_class_label]}\n'


            prediction_equals_manual = f'prediction_equals_manual = {manual_class == class_label}\n'
            prediction_equals_manual_st = f'prediction_equals_manual = {manual_class == st_class_label}\n'
            meta = open(data_path / 'clem_zfish1'/ 'functionally_imaged' / directory / f'{directory}_metadata.txt', 'r')
            t = meta.read()
            if not t[-1:] == '\n':
                t = t + '\n'

            new_t = (t + prediction_string + correlation_test + prediction_equals_manual + prediction_string_single_trial +correlation_test_single_trial+prediction_equals_manual_st)
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
                                    'manual_assigned_class': [eval(new_t.split('\n')[7][19:])[0]],
                                    'predicted_class': [class_label],
                                    'predicted_class_single_trial': [st_class_label],
                                    'max_correlation': [ci[np.argmax(ci)]],
                                    'max_correlation_single_trial' :st_best_fit,
                                    'correlation_mc':np.corrcoef(PD[40:-40], regressors[2][20:-20])[0, 1],
                                    'correlation_dt':np.corrcoef(PD[40:-40], regressors[1][20:-20])[0, 1],
                                    'correlation_i':np.corrcoef(PD[40:-40], regressors[0][20:-20])[0, 1],
                                    'passed_correlation': [ci[np.argmax(ci)] > custom_cutoff[class_label]],
                                    'passed_correlation_single_trial': [st_best_fit > custom_cutoff[st_class_label]],
                                    'manual_matches_regressor': [eval(new_t.split('\n')[7][19:])[0].replace(" ","_") == class_label],
                                    'manual_matches_regressor_St': [eval(new_t.split('\n')[7][19:])[0].replace(" ", "_") == st_class_label],
                                    'modality': 'clem',
                                    'reliability':rel,
                                    'cronbach': ca,
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
    print(f'Without threshold: {df["manual_matches_regressor"].sum()/df.shape[0]}')
    lll = df[df['passed_correlation']]
    print(f'With threshold: {lll["manual_matches_regressor"].sum() / lll.shape[0]}')


    def generate_intervals_with_min_delta(length, min_delta=1):
        intervals = []
        for start in range(length):
            for end in range(start + min_delta, length):
                intervals.append((start, end))
        return intervals



    df['manual_assigned_class'] = df['manual_assigned_class'].apply(lambda x: x.replace(" ", "_"))


    sub_mot_int = df.loc[(df['manual_assigned_class'].isin(['motor_command','motor command',"integrator"]))&(df['manual_matches_regressor']),:]
    sub_mot_int['manual_assigned_class'] = sub_mot_int['manual_assigned_class'].apply(lambda x: x.replace(" ","_"))


    all_PD = np.stack(df.PD.to_numpy())

    for i in range(all_PD.shape[0]):
        for i2 in range(all_PD[i].shape[0]):
            all_PD[i][::-1]
            if np.isnan(all_PD[i][::-1][i2]):
                all_PD[i][::-1][i2] = all_PD[i][::-1][i2-1]
    all_PD = (all_PD - np.nanmin(all_PD, axis=1)[:, np.newaxis]) / (np.nanmax(all_PD, axis=1)[:, np.newaxis] - np.nanmin(all_PD, axis=1)[:, np.newaxis])

    # Determine optimal cluster size
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
    from sklearn.mixture import GaussianMixture
    silhouette_scores = []
    wcss = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    bic_scores = []
    K = range(2, 20)

    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(all_PD)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(all_PD, kmeans.labels_)
        silhouette_scores.append(score)
        calinski_harabasz_scores.append(calinski_harabasz_score(all_PD, kmeans.labels_))
        davies_bouldin_scores.append(davies_bouldin_score(all_PD, kmeans.labels_))
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(all_PD)
        bic_scores.append(gmm.bic(all_PD))

    # Plotting the Elbow Method (WCSS vs. Number of Clusters)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 12))

    # WCSS plot
    plt.subplot(3, 2, 1)
    plt.plot(K, wcss, 'bo-')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('WCSS (Inertia)')
    plt.title('Elbow Method for Optimal k')

    # Silhouette Score plot
    plt.subplot(3, 2, 2)
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')

    # Calinski-Harabasz Index plot
    plt.subplot(3, 2, 3)  # subplot index starts from 1, not 0
    plt.plot(K, calinski_harabasz_scores, 'bo-')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Calinski-Harabasz Index')
    plt.title('Calinski-Harabasz Index')

    # Davies-Bouldin Index plot
    plt.subplot(3, 2, 4)  # subplot index starts from 1, not 0
    plt.plot(K, davies_bouldin_scores, 'bo-')
    plt.xlabel('Number of clusters, k')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Davies-Bouldin Index')

    # BIC plot (assuming bic_scores is for BIC)
    plt.subplot(3, 2, 5)  # Reuse the last subplot (2, 2, 4)
    plt.plot(K, bic_scores, 'bo-')  # Overplot on the same subplot
    plt.xlabel('Number of clusters, k')
    plt.ylabel('BIC (Gaussian Mixture Model)')
    plt.title('BIC for GMM')

    plt.tight_layout()  # Adjust spacing between plots for better readability
    # plt.show()

    #Kmeans clustering
    n_clusters =4
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(all_PD)
    label2class = {0:'integrator',1:'dynamic_threshold',2:'motor_command',3:'integrator'}
    # label2class = {2:'motor_command',1:'dynamic_threshold',0:'integrator'}
    for i in range(kmeans.cluster_centers_.shape[0]):
        plt.plot(kmeans.cluster_centers_[i],label=i)
        ci = [np.corrcoef(all_PD[i][20:-20], regressors[0][20:-20])[0, 1],
              np.corrcoef(all_PD[i][20:-20], regressors[1][20:-20])[0, 1],
              np.corrcoef(all_PD[i][20:-20], regressors[2][20:-20])[0, 1]]

    print(label2class)
    plt.legend()
    # plt.show()

    #check if cluster 3 rather belongs to cluster 0 or 1

    df['kmeans_labels'] = [label2class[x] for x in kmeans.labels_]
    df['kmeans_labels_int'] = kmeans.labels_
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

    kk = df.loc[:,['cell_name','manual_assigned_class','kmeans_labels','kmeans_labels_int']]
    kk['match'] = kk['manual_assigned_class'] == kk['kmeans_labels']
    kk['cell_name'] = kk['cell_name'].apply(lambda x: x[12:] if 'cell' in x else x)
    plt.clf()
    fig, ax = plt.subplots(n_clusters,1,figsize=(5,3.5*n_clusters))
    for i in range(kmeans.cluster_centers_.shape[0]):
        ax[i].title.set_text(f'cluster {i}')
        ax[i].plot(all_PD[kmeans.labels_ == i, :].T)
    plt.show()

    from scipy.stats import norm, kstest

    extra_analysis = True
    if extra_analysis:
        cluster3_rel = df.loc[(df['kmeans_labels_int'] == 3) & (df['manual_assigned_class'] != 'nan'), 'reliability'].to_numpy()
        cluster0_rel = df.loc[(df['kmeans_labels_int'] == 0) & (df['manual_assigned_class'] != 'nan'), 'reliability'].to_numpy()
        cluster1_rel = df.loc[(df['kmeans_labels_int'] == 1) & (df['manual_assigned_class'] != 'nan'), 'reliability'].to_numpy()
        cluster2_rel = df.loc[(df['kmeans_labels_int'] == 2) & (df['manual_assigned_class'] != 'nan'), 'reliability'].to_numpy()

        mu3, std3 = norm.fit(cluster3_rel)
        mu2, std2 = norm.fit(cluster2_rel)
        mu0, std0 = norm.fit(cluster0_rel)
        mu1, std1 = norm.fit(cluster1_rel)

        x = np.linspace(min(np.min(cluster3_rel), np.min(cluster0_rel), np.min(cluster1_rel), np.min(cluster2_rel)) - 1,
                        max(np.max(cluster3_rel), np.max(cluster0_rel), np.max(cluster1_rel),np.max(cluster2_rel)) + 1, 1000)
        cluster3_pdf = norm.pdf(x, mu3, std3)
        cluster0_pdf=norm.pdf(x, mu0, std0)
        cluster1_pdf =  norm.pdf(x, mu1, std1)
        cluster2_pdf = norm.pdf(x, mu2, std2)



        plt.figure(figsize=(10, 6))
        plt.plot(x, cluster0_pdf, label='MC',  linewidth=2)
        plt.plot(x, cluster1_pdf, label='DT',  linewidth=2)
        plt.plot(x, cluster2_pdf, label='I1',  linewidth=2)
        plt.plot(x, cluster3_pdf, label='I2', linewidth=2)
        plt.legend()
        plt.title('PDFs of Clusters')
        plt.xlabel('Reliability')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)
        plt.show()




        ks_stat0, p_value0 = kstest(cluster3_rel, lambda x: norm.cdf(x, mu0, std0))
        ks_stat1, p_value1 = kstest(cluster3_rel, lambda x: norm.cdf(x, mu1, std1))
        print('ks_stat(3vs0)', ks_stat0, '\nks_stat(3vs1)', ks_stat1, '\np_valu(3vs0)', p_value0, '\np_value(3vs1)', p_value1)

        print('Probability distribution 3 vs 0', np.sum(norm.logpdf(cluster3_rel, mu0, std0)),'\nProbability distribution 3 vs 1', np.sum(norm.logpdf(cluster3_rel, mu1, std1)))
        plt.figure(figsize=(4, 10))

        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor='blue', edgecolor='k',label='MC'),
                           Patch(facecolor='orange', edgecolor='k',label='DT'),
                           Patch(facecolor='green', edgecolor='k',label='I1'),
                           Patch(facecolor='red', edgecolor='k',label='I2')]

        import seaborn
        seaborn.boxplot(x=1, y=cluster0_rel, width=1,color='blue')
        seaborn.boxplot(x=2, y=cluster1_rel, width=1,color='orange')
        seaborn.boxplot(x=3, y=cluster2_rel, width=1,color='green')
        seaborn.boxplot(x=4, y=cluster3_rel, width=1,color='red')
        plt.ylim(-2,7)
        plt.legend(handles=legend_elements)
        plt.show()






    from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
    import navis
    import plotly
    path_to_data = get_base_path()
    brain_meshes = load_brs(path_to_data, 'raphe')

    em_pa_cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem', 'pa'], load_repaired=True)




    for i in range(kmeans.cluster_centers_.shape[0]):
        em_pa_cells.loc[em_pa_cells['cell_name'].isin(kk.loc[kk['kmeans_labels_int']==i,'cell_name']),'kmeans_functional_label'] = int(i)
        em_pa_cells.loc[em_pa_cells['cell_name'].isin(kk.loc[kk['kmeans_labels_int'] == i, 'cell_name']), 'kmeans_functional_label_str'] = label2class[i]

    accepted_function = ['integrator', 'motor_command', 'dynamic_threshold', 'integrator','dynamic threshold','motor command']

    for i,cell in em_pa_cells.iterrows():
        if cell.cell_name == 'cell_576460752652865344':
            pass
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

    #reliability in subset
    class2location = {'integrator': -0.2, 'motor_command': 0, 'dynamic_threshold': 0.2}
    x_modifier = {0: -0.2, 1: -0.1, 2: +0.1,3:0.2}



    plt.figure(dpi=300)
    for i in range(len([x for x in df['kmeans_labels_int'].unique() if x!='nan'])):
        temp = df.loc[(df['kmeans_labels_int'] == i) & (df['manual_assigned_class'] != 'nan'), :]
        plt.scatter([i + class2location[x] for x in temp['manual_assigned_class']], temp['reliability'])
    plt.xticks([-0.2, 0, 0.2, 0.8, 1, 1.2,1.8,2,2.2,2.8,3,3.2],['I','MC','DT']*4)
    plt.xlim(-0.4,3.4)
    plt.ylabel('reliability')
    plt.xlabel('Manual assigned classes across kmeans clusters')

    #plt.show()
    plt.clf()
    label2class

    color_dict = {
        "integrator": '#e84d8ab3',
        "dynamic_threshold": '#64c5ebb3',
        "motor_command": '#7f58afb3',
        'neg control': "#a8c256b3"
    }
    in_legend = []
    np.random.seed(1)
    for i,item in df.iterrows():
        offset=-0.2+np.random.choice(np.arange(-0.075,0.075,0.01))
        marker = 'o'
        if item.imaging_modality == 'photoactivation':
            offset = 0.2+np.random.choice(np.arange(-0.075,0.075,0.01))
            marker = 's'
        if label2class[item.kmeans_labels_int] + " " + item.imaging_modality not in in_legend:
            plt.scatter(item.kmeans_labels_int+offset,item.reliability,c=color_dict[label2class[item.kmeans_labels_int]],label = label2class[item.kmeans_labels_int] + " " + item.imaging_modality,marker=marker)
            in_legend.append(label2class[item.kmeans_labels_int] + " " + item.imaging_modality)
        else:
            plt.scatter(item.kmeans_labels_int + offset, item.reliability, c=color_dict[label2class[item.kmeans_labels_int]],marker=marker)
        legend = plt.legend(frameon=False,fontsize='xx-small')
    plt.ylim(-2,7)
    plt.show()

    #save kmeans regressor
    os.makedirs(data_path / 'make_figures_FK_output' / 'functional_analysis', exist_ok=True)
    np.save(data_path / 'make_figures_FK_output' / 'functional_analysis'/'new_regressors.npy',kmeans.cluster_centers_)
    em_pa_cells.loc[:,['cell_name','functional_id','kmeans_functional_label_str']].to_excel('assignment_.xlsx')

    np.save('kmeans_regressors.npy',kmeans.cluster_centers_,)
