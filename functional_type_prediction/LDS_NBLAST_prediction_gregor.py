from hindbrain_structure_function.functional_type_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.functional_type_prediction.LDS_predict_jon_cells import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from cellpose import denoise
from cellpose import denoise
import cv2
import pandas as pd
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
from hindbrain_structure_function.visualization.FK_tools.load_pa_table import *
import re
from datetime import datetime
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.fragment_neurite import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.slack_bot import *
from matplotlib import colors
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette
from hindbrain_structure_function.functional_type_prediction.FK_tools.slack_bot import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm
from hindbrain_structure_function.functional_type_prediction.FK_tools.branching_angle_calculator import *
import copy
import os
from hindbrain_structure_function.functional_type_prediction.nblast_matrix_navis import *
import matplotlib.pyplot as plt
import scipy
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.functional_type_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.make_dendrogramms import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from hindbrain_structure_function.functional_type_prediction.FK_tools.find_branches import *
import winsound
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import plotly
import matplotlib
# matplotlib.use('TkAgg')
from tqdm import tqdm
from sklearn.svm import OneClassSVM

if __name__ == '__main__':

    #set variables
    np.set_printoptions(suppress=True)
    path_to_data = get_base_path()
    brain_meshes = load_brs(path_to_data, 'raphe')
    width_brain = 495.56


    #load cells

    predict_cells_em = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['em'], load_repaired=True)
    predict_cells_em = predict_cells_em.reset_index(drop=True)

    train_cells1 = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem', 'pa'], load_repaired=True)
    train_cells2 = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['prediction_project'],
                                                             load_repaired=True)
    train_cells = pd.concat([train_cells1,train_cells2])
    train_cells = train_cells.drop_duplicates(keep='first', inplace=False,subset='cell_name')
    train_cells_no_function =  train_cells.loc[(train_cells.function == 'nan'), :]
    train_cells = train_cells.loc[(train_cells.function != 'nan'), :]
    train_cells = train_cells.loc[(~train_cells.function.isna()), :]
    train_cells = train_cells.reset_index(drop=True)





    for i,cell in predict_cells_em.iterrows():


        if (cell.swc.nodes.x.to_numpy()>=(width_brain/2)).any():
            predict_cells_em.loc[i,'morphology'] = 'contralateral'
        else:
            predict_cells_em.loc[i, 'morphology'] = 'ipsilateral'


    print('\nFINISHED LOADING CELLS\n')


    #calculate metrics
    print('PREDICT')
    calculate_metric(predict_cells_em,'predict_complete_em',path_to_data,force_new=False,train_or_predict='predict')
    print('TRAIN')
    calculate_metric(train_cells, 'train_complete',path_to_data, force_new=False,train_or_predict='train')
    print('\nFINISHED CALCULATING METRICS\n')


    #Load data to train model
    features_train, labels_train, labels_imaging_modality_train, column_labels_train, df_train = load_train_data(path_to_data,file='train_complete')
    
    #load predict data
    features_predict_em, labels_imaging_modality_predict_em, column_labels_predict_em, df_predict_em,cell_names_em = load_predict_data(path_to_data,'predict_complete_em')

    #find reduced features
    # _, reduced_features_index_train = determine_important_features_RFECV(features_train, labels_train, column_labels_train, scoring='roc_auc_ovo')
    _, reduced_features_index, _ = determine_important_features(features_train, labels_train, column_labels_train, return_collection_coef_matrix=True)

    #select cells via nblast
    smat_fish = load_zebrafish_nblast_matrix(return_smat_obj=True, prune=False, modalities=['clem', 'pa'])

    #nb = nblast_two_groups(train_cells,train_cells,shift_neurons=False)
    nb = nblast_two_groups_custom_matrix(train_cells,train_cells,custom_matrix=smat_fish,shift_neurons=False)
    #nb = nblast_two_groups(train_cells.loc[train_cells['imaging_modality'] == 'clem', :], train_cells.loc[train_cells['imaging_modality'] == 'clem', :], shift_neurons=False)

    aaa = navis.nbl.extract_matches(nb, 2)
    #nb = nblast_two_groups(train_cells,predict_cells_em,shift_neurons=False)
    nb = nblast_two_groups_custom_matrix(train_cells, predict_cells_em, custom_matrix=smat_fish, shift_neurons=False)
    #nb = nblast_two_groups(train_cells.loc[train_cells['imaging_modality'] == 'clem', :], predict_cells_em, shift_neurons=False)
    bbb = navis.nbl.extract_matches(nb.T, 2)

    cutoff= np.percentile(list(aaa.score_2),10)

    #subset based on nblast
    subset_predict_cells = list(bbb.loc[bbb['score_1']>=cutoff,'id'])
    acronym_dict={'dynamic threshold':"DT",'integrator contralateral':"CI",'integrator ipsilateral':"II",'motor command':"MC"}
    color_dict = {
            "integrator ipsilateral": '#feb326b3',
            "integrator contralateral": '#e84d8ab3',
            "dynamic threshold": '#64c5ebb3',
            "motor command": '#7f58afb3',
        }


    #both reduced
    solver = 'lsqr'
    shrinkage = 'auto'
    priors = [len(labels_train[labels_train== x]) / len(labels_train) for x in np.unique(labels_train)]
    clf_both = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors)
    clf_both.fit(features_train[:,reduced_features_index], labels_train.flatten())
    OCSVM =  OneClassSVM(gamma='scale',kernel='rbf').fit(features_train[:,reduced_features_index])
    IF = IsolationForest(contamination=0.0001,random_state=42).fit(features_train[:,reduced_features_index])
    LOF = LocalOutlierFactor(n_neighbors=10,novelty=True).fit(features_train[:,reduced_features_index])


    color_list = []
    swc_list = []

    for i,cell_name in zip(range(features_predict_em.shape[0]),cell_names_em):
        #prediction
        y_prob = clf_both.predict_proba(features_predict_em[i, reduced_features_index].reshape(1, -1))[0]
        y_pred = clf_both.predict(features_predict_em[i,reduced_features_index].reshape(1, -1))[0]

        # outlier/novelty detection
        outlier_ocsvm = OCSVM.predict(features_predict_em[i, reduced_features_index].reshape(1, -1))[0] == 1
        outlier_IF = IF.predict(features_predict_em[i, reduced_features_index].reshape(1, -1))[0] == 1
        outlier_LOF = LOF.predict(features_predict_em[i, reduced_features_index].reshape(1, -1))[0] == 1


        temp_title = (f'{acronym_dict[clf_both.classes_[0]]}: {"{:.5f}".format(y_prob[0]*100)}%<br>'
                      f'{acronym_dict[clf_both.classes_[1]]}: {"{:.5f}".format(y_prob[1]*100)}%<br>'
                      f'{acronym_dict[clf_both.classes_[2]]}: {"{:.5f}".format(y_prob[2]*100)}%<br>'
                      f'{acronym_dict[clf_both.classes_[3]]}: {"{:.5f}".format(y_prob[3]*100)}%')
        temp_prediction_string = temp_title = (f'{acronym_dict[clf_both.classes_[0]]}:{"{:.5f}".format(y_prob[0])} '
                                               f'{acronym_dict[clf_both.classes_[1]]}:{"{:.5f}".format(y_prob[1])} '
                                               f'{acronym_dict[clf_both.classes_[2]]}:{"{:.5f}".format(y_prob[2])} '
                                               f'{acronym_dict[clf_both.classes_[3]]}:{"{:.5f}".format(y_prob[3])}')


        prediction_string = f'prediction = "{temp_prediction_string}"\n'
        nblast_test_string = f'nblast_test_passed = {cell_name in subset_predict_cells}\n'
        probability_passed_string = f'probability_test_passed = {(y_prob>0.7).any()}\n'


        meta_paths = predict_cells_em.loc[predict_cells_em['cell_name']==cell_name,'metadata_path']
        for meta_path in meta_paths:
            f = open(meta_path, 'r')
            t = f.read()
            new_t = t + prediction_string + nblast_test_string + probability_passed_string
            f.close()

            f = open(str(meta_path)[:-4] + "_with_prediction_reduced_features.txt", 'w')
            f.write(new_t)
            f.close()




        if (np.max(y_prob)>0.7) and cell_name in subset_predict_cells:
            temp_swc = predict_cells_em.loc[predict_cells_em['cell_name']==cell_name,'swc'].iloc[0]
            temp_node_id = temp_swc.nodes.loc[temp_swc.nodes.type=='root','node_id'].iloc[0]
            temp_swc.soma = temp_node_id
            fig = navis.plot3d(predict_cells_em.loc[predict_cells_em['cell_name']==cell_name,'swc'].iloc[0], backend='plotly',
                               width=1920, height=1080, hover_name=True, alpha=1,title=temp_title)
            fig = navis.plot3d(brain_meshes, backend='plotly', fig=fig,
                               width=1920, height=1080, hover_name=True,title=temp_title)
            fig.update_layout(
                scene={
                    'xaxis': {'autorange': 'reversed'},  # reverse !!!
                    'yaxis': {'autorange': True},

                    'zaxis': {'autorange': True},
                    'aspectmode': "data",
                    'aspectratio': {"x": 1, "y": 1, "z": 1}},
                title = dict(text=temp_title, font=dict(size=20), automargin=True, yref='paper')
            )
            os.makedirs(path_to_data / 'make_figures_FK_output'/'LDS_NBLAST_predictions_reduced_features_em',exist_ok=True)
            temp_file_name= path_to_data / 'make_figures_FK_output'/'LDS_NBLAST_predictions_reduced_features_em'/f"{acronym_dict[y_pred]}_{cell_name}.html"
            plotly.offline.plot(fig, filename=str(temp_file_name), auto_open=False, auto_play=False)
            color_list.append(color_dict[y_pred])
            swc_list.append(predict_cells_em.loc[predict_cells_em['cell_name']==cell_name,'swc'].iloc[0])

    if len(color_list) > 0:
        fig = navis.plot3d(swc_list, backend='plotly',colors=color_list,
                           width=1920, height=1080, hover_name=True, alpha=1, title=temp_title)
        fig = navis.plot3d(brain_meshes, backend='plotly', fig=fig,
                           width=1920, height=1080, hover_name=True, title=temp_title)
        fig.update_layout(
            scene={
                'xaxis': {'autorange': 'reversed'},  # reverse !!!
                'yaxis': {'autorange': True},

                'zaxis': {'autorange': True},
                'aspectmode': "data",
                'aspectratio': {"x": 1, "y": 1, "z": 1}},
            title=dict(text='reduced features', font=dict(size=20), automargin=True, yref='paper')
        )
        os.makedirs(path_to_data / 'make_figures_FK_output' / 'LDS_NBLAST_predictions_em', exist_ok=True)
        temp_file_name = path_to_data / 'make_figures_FK_output' / 'LDS_NBLAST_predictions_reduced_features_em' / f"all_cells_reduced_em.html"
        plotly.offline.plot(fig, filename=str(temp_file_name), auto_open=True, auto_play=False)


    #both
    solver = 'lsqr'
    shrinkage = 'auto'
    priors = [len(labels_train[labels_train== x]) / len(labels_train) for x in np.unique(labels_train)]
    clf_both = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors)
    clf_both.fit(features_train, labels_train.flatten())
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    OCSVM =  OneClassSVM(gamma='scale',kernel='rbf').fit(features_train)
    IF = IsolationForest(contamination=0.0001,random_state=42).fit(features_train)
    LOF = LocalOutlierFactor(n_neighbors=10,novelty=True).fit(features_train)



    color_list = []
    swc_list = []

    for i,cell_name in zip(range(features_predict_em.shape[0]),cell_names_em):
        #prediction
        y_prob = clf_both.predict_proba(features_predict_em[i, :].reshape(1, -1))[0]
        y_pred = clf_both.predict(features_predict_em[i, :].reshape(1, -1))[0]

        # outlier/novelty detection
        outlier_ocsvm = OCSVM.predict(features_predict_em[i, :].reshape(1, -1))[0] == 1
        outlier_IF = IF.predict(features_predict_em[i, :].reshape(1, -1))[0] == 1
        outlier_LOF = LOF.predict(features_predict_em[i, :].reshape(1, -1))[0] == 1






        temp_title = (f'{acronym_dict[clf_both.classes_[0]]}: {"{:.5f}".format(y_prob[0]*100)}%<br>'
                      f'{acronym_dict[clf_both.classes_[1]]}: {"{:.5f}".format(y_prob[1]*100)}%<br>'
                      f'{acronym_dict[clf_both.classes_[2]]}: {"{:.5f}".format(y_prob[2]*100)}%<br>'
                      f'{acronym_dict[clf_both.classes_[3]]}: {"{:.5f}".format(y_prob[3]*100)}%')
        temp_prediction_string = temp_title = (f'{acronym_dict[clf_both.classes_[0]]}:{"{:.5f}".format(y_prob[0])} '
                                               f'{acronym_dict[clf_both.classes_[1]]}:{"{:.5f}".format(y_prob[1])} '
                                               f'{acronym_dict[clf_both.classes_[2]]}:{"{:.5f}".format(y_prob[2])} '
                                               f'{acronym_dict[clf_both.classes_[3]]}:{"{:.5f}".format(y_prob[3])}')


        prediction_string = f'prediction = "{temp_prediction_string}"\n'
        nblast_test_string = f'nblast_test_passed = {cell_name in subset_predict_cells}\n'
        probability_passed_string = f'probability_test_passed = {(y_prob>0.7).any()}\n'


        meta_paths = predict_cells_em.loc[predict_cells_em['cell_name']==cell_name,'metadata_path']
        for meta_path in meta_paths:
            f = open(meta_path, 'r')
            t = f.read()
            new_t = t + prediction_string + nblast_test_string + probability_passed_string
            f.close()

            f = open(str(meta_path)[:-4] + "_with_prediction.txt", 'w')
            f.write(new_t)
            f.close()


        if (np.max(y_prob)>0.7) and cell_name in subset_predict_cells:# and outlier_IF and outlier_LOF:
            temp_swc = predict_cells_em.loc[predict_cells_em['cell_name']==cell_name,'swc'].iloc[0]
            temp_node_id = temp_swc.nodes.loc[temp_swc.nodes.type=='root','node_id'].iloc[0]
            temp_swc.soma = temp_node_id

            fig = navis.plot3d(predict_cells_em.loc[predict_cells_em['cell_name']==cell_name,'swc'].iloc[0], backend='plotly',
                               width=1920, height=1080, hover_name=True, alpha=1,title=temp_title)
            fig = navis.plot3d(brain_meshes, backend='plotly', fig=fig,
                               width=1920, height=1080, hover_name=True,title=temp_title)
            fig.update_layout(
                scene={
                    'xaxis': {'autorange': 'reversed'},  # reverse !!!
                    'yaxis': {'autorange': True},

                    'zaxis': {'autorange': True},
                    'aspectmode': "data",
                    'aspectratio': {"x": 1, "y": 1, "z": 1}},
                title = dict(text=temp_title, font=dict(size=20), automargin=True, yref='paper')
            )
            os.makedirs(path_to_data / 'make_figures_FK_output'/'LDS_NBLAST_predictions_em',exist_ok=True)
            temp_file_name= path_to_data / 'make_figures_FK_output'/'LDS_NBLAST_predictions_em'/f"{acronym_dict[y_pred]}_{cell_name}.html"
            plotly.offline.plot(fig, filename=str(temp_file_name), auto_open=False, auto_play=False)
            color_list.append(color_dict[y_pred])
            swc_list.append(predict_cells_em.loc[predict_cells_em['cell_name']==cell_name,'swc'].iloc[0])

    if len(color_list) > 0:
        fig = navis.plot3d(swc_list, backend='plotly',colors=color_list,
                           width=1920, height=1080, hover_name=True, alpha=1, title=temp_title)
        fig = navis.plot3d(brain_meshes, backend='plotly', fig=fig,
                           width=1920, height=1080, hover_name=True, title=temp_title)
        fig.update_layout(
            scene={
                'xaxis': {'autorange': 'reversed'},  # reverse !!!
                'yaxis': {'autorange': True},

                'zaxis': {'autorange': True},
                'aspectmode': "data",
                'aspectratio': {"x": 1, "y": 1, "z": 1}},
            title=dict(text='all features', font=dict(size=20), automargin=True, yref='paper')
        )
        os.makedirs(path_to_data / 'make_figures_FK_output' / 'LDS_NBLAST_predictions_em', exist_ok=True)
        temp_file_name = path_to_data / 'make_figures_FK_output' / 'LDS_NBLAST_predictions_em' / f"all_cells_em.html"
        plotly.offline.plot(fig, filename=str(temp_file_name), auto_open=True, auto_play=False)
