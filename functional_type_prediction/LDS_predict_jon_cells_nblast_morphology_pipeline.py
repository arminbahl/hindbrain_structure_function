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

import matplotlib.pyplot as plt
import scipy
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.functional_type_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.make_dendrogramms import *
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

if __name__ == '__main__':

    #set variables
    np.set_printoptions(suppress=True)
    path_to_data = get_base_path()
    brain_meshes = load_brs(path_to_data, 'raphe')
    width_brain = 495.56


    #load cells
    train_cells1 = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem', 'pa'], load_repaired=True)
    train_cells2 = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['prediction_project'],
                                                             load_repaired=True)
    train_cells = pd.concat([train_cells1,train_cells2])
    train_cells = train_cells.drop_duplicates(keep='first', inplace=False,subset='cell_name')
    train_cells_no_function =  train_cells.loc[(train_cells.function == 'nan'), :]
    train_cells = train_cells.loc[(train_cells.function != 'nan'), :]
    train_cells = train_cells.loc[(~train_cells.function.isna()), :]
    train_cells = train_cells.reset_index(drop=True)

    predict_cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem_predict'], load_repaired=True)
    predict_cells = pd.concat([predict_cells,train_cells_no_function])
    predict_cells = predict_cells.drop_duplicates(keep='first', inplace=False, subset='cell_name')
    predict_cells = predict_cells.loc[(~predict_cells['cell_name'].isin(train_cells.cell_name)),:]
    predict_cells = predict_cells.loc[predict_cells['type']!='axon',:]
    predict_cells = predict_cells.reset_index(drop=True)
    for i,cell in predict_cells.iterrows():


        if (cell.swc.nodes.x.to_numpy()>=(width_brain/2)).any():
            predict_cells.loc[i,'morphology'] = 'contralateral'
        else:
            predict_cells.loc[i, 'morphology'] = 'ipsilateral'


    print('\nFINISHED LOADING CELLS\n')
    # predict_cells['check_morphology'] = predict_cells.apply(lambda x: (x['swc'].nodes.x.to_numpy() >= (width_brain / 2)).any(), axis=1)

    #calculate metrics
    calculate_metric(predict_cells,'predict_complete',path_to_data,force_new=True,train_or_predict='predict')
    calculate_metric(train_cells, 'train_complete',path_to_data, force_new=False,train_or_predict='train')
    print('\nFINISHED CALCULATING METRICS\n')


    #Load data to train model
    features_train, labels_train, labels_imaging_modality_train, column_labels_train, df_train = load_train_data(path_to_data,file='train_complete')
    
    #load predict data
    features_predict, labels_imaging_modality_predict, column_labels_predict, df_predict,cell_names = load_predict_data(path_to_data,'predict_complete')

    #find reduced features
    reduced_features_train, reduced_features_index_train = determine_important_features_RFECV(features_train, labels_train, column_labels_train, scoring='roc_auc_ovo')
    reduced_features, reduced_features_index, collection_coef_matrix = determine_important_features(features_train, labels_train, column_labels_train, return_collection_coef_matrix=True)

    #select cells via nblast
    nb = nblast_two_groups(train_cells,train_cells,shift_neurons=False)
    aaa = navis.nbl.extract_matches(nb, 2)
    nb = nblast_two_groups(train_cells,predict_cells,shift_neurons=False)
    bbb = navis.nbl.extract_matches(nb.T, 2)

    cutoff= np.percentile(list(aaa.score_2),10)

    #subset based on nblast
    subset_predict_cells = list(bbb.loc[bbb['score_1']>=cutoff,'id'])
    bool_subset_nblast = list(df_predict.cell_name.isin(subset_predict_cells))
    
    features_predict_subset_nblast = features_predict[bool_subset_nblast,:]
    labels_imaging_modality_predict_subset_nblast = np.empty(features_predict_subset_nblast.shape[0]).astype('<U4')
    labels_imaging_modality_predict_subset_nblast[:] = 'clem'
    df_predict_subset_nblast = df_predict.loc[bool_subset_nblast,:]
    cell_names_subset_nblast = df_predict.loc[bool_subset_nblast,'cell_name']

    #subset based on morphology

    #ipsi predict
    bool_ipsi = list(df_predict_subset_nblast['morphology_clone'] == 'ipsilateral')
    features_predict_subset_ipsi = features_predict_subset_nblast[bool_ipsi,:]
    labels_imaging_modality_predict_ispi= np.empty(features_predict_subset_ipsi.shape[0]).astype('<U4')
    labels_imaging_modality_predict_ispi[:] = 'clem'
    df_predict_subset_ipsi = df_predict_subset_nblast.loc[bool_ipsi,:]
    cell_names_subset_ipsi = df_predict_subset_nblast.loc[bool_ipsi,'cell_name']

    #contra predict
    bool_contra = [not x for x in bool_ipsi]
    features_predict_subset_contra = features_predict_subset_nblast[bool_contra,:]
    labels_imaging_modality_predict_contra = np.empty(features_predict_subset_contra.shape[0]).astype('<U4')
    labels_imaging_modality_predict_contra[:] = 'clem'
    df_predict_subset_contra = df_predict_subset_nblast.loc[bool_contra,:]
    cell_names_subset_contra = df_predict_subset_nblast.loc[bool_contra,'cell_name']

    #ipsi train
    bool_ipsi = list(df_train['morphology_clone'] == 'ipsilateral')
    features_train_subset_ipsi = features_train[bool_ipsi,:]
    labels_train_subset_ipsi = labels_train[bool_ipsi]
    labels_imaging_modality_train_subset_ipsi = df_train.loc[bool_ipsi,'imaging_modality'].to_numpy()
    df_train_subset_ipsi = df_train.loc[bool_ipsi,:]

    #contra train
    bool_contra = list(df_train['morphology_clone'] == 'contralateral')
    features_train_subset_contra = features_train[bool_contra,:]
    labels_train_subset_contra = labels_train[bool_contra]
    labels_imaging_modality_train_subset_contra = df_train.loc[bool_contra,'imaging_modality'].to_numpy()
    df_train_subset_contra = df_train.loc[bool_contra,:]





    #prediction
    solver = 'lsqr'
    shrinkage = 'auto'
    priors_ipsi = [len(labels_train_subset_ipsi[labels_train_subset_ipsi == x]) / len(labels_train_subset_ipsi) for x in np.unique(labels_train_subset_ipsi)]
    clf_ipsi = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors_ipsi)
    clf_ipsi.fit(features_train_subset_ipsi, labels_train_subset_ipsi.flatten())
    
    
    solver = 'lsqr'
    shrinkage = 'auto'
    priors_contra = [len(labels_train_subset_contra[labels_train_subset_contra == x]) / len(labels_train_subset_contra) for x in np.unique(labels_train_subset_contra)]
    clf_contra = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors_contra)
    clf_contra.fit(features_train_subset_contra, labels_train_subset_contra.flatten())


    for i,cell_name in zip(range(features_predict_subset_contra.shape[0]),cell_names_subset_contra):
        print(predict_cells.loc[predict_cells['cell_name']==cell_name,'morphology'].iloc[0])
        y_pred = clf_contra.predict(features_predict_subset_contra[i,:].reshape(1,-1))
        y_prob = clf_contra.predict_proba(features_predict_subset_contra[i,:].reshape(1,-1))[0]

        # print(cell_name,'\n',clf_contra.classes_[0],np.round(y_prob[0],4),clf_contra.classes_[1],np.round(y_prob[1],4),clf_contra.classes_[2],np.round(y_prob[2],4),'\n')
        temp_title = f'{clf_contra.classes_[0]}:{np.round(y_prob[0],4)}\n{clf_contra.classes_[1]}: {np.round(y_prob[1], 4)}\n{clf_contra.classes_[2]}:{np.round(y_prob[2],4)}'
        if (y_prob>0.9).any():
            temp_swc = predict_cells.loc[predict_cells['cell_name']==cell_name,'swc'].iloc[0]
            temp_node_id = temp_swc.nodes.loc[temp_swc.nodes.type=='root','node_id'].iloc[0]
            temp_swc.soma = temp_node_id
            fig = navis.plot3d(predict_cells.loc[predict_cells['cell_name']==cell_name,'swc'].iloc[0], backend='plotly',
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
            plotly.offline.plot(fig, filename=f"{cell_name}.html", auto_open=True, auto_play=False)
            
    
    for i,cell_name in zip(range(features_predict_subset_ipsi.shape[0]),cell_names_subset_ipsi):
        y_pred = clf_ipsi.predict(features_predict_subset_ipsi[i,:].reshape(1,-1))
        y_prob = clf_ipsi.predict_proba(features_predict_subset_ipsi[i,:].reshape(1,-1))[0]

        # print(cell_name,'\n',clf_ipsi.classes_[0],np.round(y_prob[0],4),clf_ipsi.classes_[1],np.round(y_prob[1],4),clf_ipsi.classes_[2],np.round(y_prob[2],4),'\n')
        temp_title = f'{clf_ipsi.classes_[0]}:{np.round(y_prob[0],4)}\n{clf_ipsi.classes_[1]}: {np.round(y_prob[1], 4)}\n{clf_ipsi.classes_[2]}:{np.round(y_prob[2],4)}'
        if (y_prob>0.9).any():
            temp_swc = predict_cells.loc[predict_cells['cell_name']==cell_name,'swc'].iloc[0]
            temp_node_id = temp_swc.nodes.loc[temp_swc.nodes.type=='root','node_id'].iloc[0]
            temp_swc.soma = temp_node_id
            fig = navis.plot3d(predict_cells.loc[predict_cells['cell_name']==cell_name,'swc'].iloc[0], backend='plotly',
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
            plotly.offline.plot(fig, filename=f"{cell_name}.html", auto_open=True, auto_play=False)

    send_slack_message(RECEIVER="Florian Kämpf", MESSAGE="LDS_predict_jon_cells_nblast_morphology_pipeline finished!")