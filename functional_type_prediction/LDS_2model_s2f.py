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

def generate_matching_plot_2model(features,labels,labels_imaging_modality,path,column_labels,morphology,cell_names,solver='lsqr',shrinkage='auto',title='prediction_plot',match_limit=0.5,return_metrics=False):
    #init variables
    prob_matrix = np.full(fill_value=0,shape=(features.shape[0],len(np.unique(labels)))).astype(float)
    pred_matrix = np.empty(shape=(features.shape[0],1),dtype='<U24')
    prediction_correct = []
    legend_elements = []
    used_labels = []
    for label in column_labels:
        used_labels.append(Patch(facecolor='white', edgecolor='white', label=label))

    classes = ['dynamic threshold', 'integrator contralateral', 'integrator ipsilateral', 'motor command']
    ipsilateral_bool = df_train.morphology_clone.to_numpy() == 'ipsilateral'
    contralateral_bool = df_train.morphology_clone.to_numpy() == 'contralateral'

    #loop over cells
    for cell_name,morph,i in zip(cell_names,morphology,range(features.shape[0])):
        if morph == 'ipsilateral':
            feat_temp = features_train[ipsilateral_bool,:]
            labels_temp = labels_train[ipsilateral_bool]
            cell_names_temp = cell_names[ipsilateral_bool]
        elif morph == 'contralateral':
            feat_temp = features_train[contralateral_bool,:]
            labels_temp = labels_train[contralateral_bool]
            cell_names_temp = cell_names[contralateral_bool]


        X_train = features[[x for x,y in zip(range(feat_temp.shape[0]),cell_names_temp) if y != cell_name]]
        X_test = features[[x for x,y in zip(range(feat_temp.shape[0]),cell_names_temp) if y == cell_name]]
        y_train = labels_temp[[x for x,y in zip(range(feat_temp.shape[0]),cell_names_temp) if y != cell_name]]
        y_test = labels_temp[[x for x,y in zip(range(feat_temp.shape[0]),cell_names_temp) if y == cell_name]]

        #create and fit lda
        priors = [len(labels_temp[labels_temp == x]) / len(labels_temp) for x in np.unique(labels_temp)]
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage,priors=priors)
        clf.fit(X_train, y_train)

        # predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[0]


        if morph == 'ipsilateral':
            y_prob = np.array([y_prob[0],0,y_prob[1],y_prob[2]])
        elif morph == 'contralateral':
            y_prob = np.array([y_prob[0],y_prob[1],0,y_prob[2]])

        prob_matrix[i,:] = y_prob


        if y_prob[np.argwhere(classes==y_pred)[0][0]]>=y_prob[np.argwhere(classes!=y_pred)].sum() and y_prob[np.argwhere(classes==y_pred)[0][0]]>=match_limit:
            prediction_correct.append((y_pred==y_test)[0])
            pred_matrix[i, :] = y_pred


    print(f'\nPredictions correct: {np.round((np.sum(prediction_correct) / features.shape[0] * 100), 2)}%'
          f'\nPredictions incorrect: {np.round((np.sum([not x for x in prediction_correct]) / features.shape[0] * 100), 2)}%'
          f'\nNo prediction: {np.round((features.shape[0] - len(prediction_correct)) / features.shape[0] * 100, 2)}%\n')
    #figure part
    if return_metrics:
        correct = np.round((np.sum(prediction_correct) / features.shape[0] * 100), 2)
        incorrect = np.round((np.sum([not x for x in prediction_correct]) / features.shape[0] * 100), 2)
        no_prediction = np.round((features.shape[0] - len(prediction_correct)) / features.shape[0] * 100, 2)
        return correct, incorrect, no_prediction
    else:
        color_dict_type = {
            "integrator ipsilateral": '#feb326b3',
            "integrator contralateral": '#e84d8ab3',
            "dynamic threshold": '#64c5ebb3',
            "motor command": '#7f58afb3',
        }


        color_dict_modality = {'clem': 'black', "photoactivation": "gray"}

        fig, ax = plt.subplots(figsize=(40, 8))

        im = ax.pcolormesh(prob_matrix.T)
        ax.plot([-1, -1], [-1, -1])

        labels_sort = np.unique(labels)
        labels_sort.sort()
        location_dict = {}
        for i, label in enumerate(labels_sort):
            location_dict[label] = i
            ax.plot([-1, -1], [0 + i, 1 + i], color=color_dict_type[label], lw=3, solid_capstyle='butt')
            temp_indices = np.argwhere(labels == label).flatten()
            ax.plot([np.min(temp_indices), np.max(temp_indices) + 1], [-0.25, -0.25], color=color_dict_type[label], lw=3, solid_capstyle='butt', alpha=1)

            if len(np.unique(labels_imaging_modality)) > 1:
                for i2, modality in enumerate(color_dict_modality.keys()):
                    temp_indices = np.argwhere((labels == label) & (labels_imaging_modality == modality))
                    ax.plot([np.min(temp_indices), np.max(temp_indices) + 1], [-0.5, -0.5], color=color_dict_modality[modality], lw=3, solid_capstyle='butt', alpha=1)
                    if not modality in [x.get_label() for x in legend_elements]:
                        legend_elements.append(Patch(facecolor=color_dict_modality[modality], edgecolor=color_dict_modality[modality], label=modality))
            if not label in [x.get_label() for x in legend_elements]:
                legend_elements.append(Patch(facecolor=color_dict_type[label], edgecolor=color_dict_type[label], label=label))


        for x, item in enumerate(pred_matrix):
            if item != '':
                y = location_dict[item[0]]
                plt.plot([x,x],[y,y+1],lw=2,color='red')
                plt.plot([x, x+1], [y+1, y + 1], lw=2, color='red')
                plt.plot([x, x + 1], [y, y ], lw=2, color='red')
                plt.plot([x+1 , x+1], [y, y + 1], lw=2, color='red')

        ax.set_yticks(np.arange(len(labels_sort)) + 0.5, [x + " prediction" for x in labels_sort])
        ax.set_xlim(-2, len(features))
        ax.set_ylim(-2, len(labels_sort))

        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        fig.colorbar(im, orientation='vertical')
        ax.set_xticks([])
        plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.1)
        savepath = path_to_data / 'make_figures_FK_output' / 'LDA_cell_type_prediction'
        os.makedirs(savepath,exist_ok=True)
        os.makedirs(savepath/'png', exist_ok=True)
        os.makedirs(savepath/'pdf', exist_ok=True)
        fig.set_dpi(450)

        first_legend = ax.legend(handles=legend_elements, frameon=False, loc=8, ncol=len(legend_elements))
        ax.add_artist(first_legend)
        second_legend = ax.legend(handles=used_labels, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(1.225, 0.975), alignment='left')
        ax.text(1.195, 1.05, f'Used features N={len(column_labels)}', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, font={'weight': 'heavy', 'size': 20, })
        ax.add_artist(second_legend)
        plt.title(title + f'\nPredictions correct: {np.round((np.sum(prediction_correct) / features.shape[0] * 100), 2)}%'
                          f'\nPredictions incorrect: {np.round((np.sum([not x for x in prediction_correct]) / features.shape[0] * 100), 2)}%'
                          f'\nNo prediction: {np.round((features.shape[0]-len(prediction_correct))/features.shape[0] * 100,2)}%\n')
        plt.savefig(savepath /'pdf'/ (title.replace('\n'," ") + ".pdf"))
        plt.savefig(savepath /'png'/ (title.replace('\n'," ")+ ".png"))

        path_to_open = savepath /'pdf'/ (title.replace('\n'," ")+ ".pdf")

        os.startfile(path_to_open)


if __name__ == "__main__":
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

    #calculate metrics
    calculate_metric(train_cells, 'train_complete', path_to_data, force_new=False, train_or_predict='train')

    #Load data to train model
    features_train, labels_train, labels_imaging_modality_train, column_labels_train, df_train = load_train_data(path_to_data,file='train_complete')

    #find reduced features
    reduced_features_train, reduced_features_index_train = determine_important_features_RFECV(features_train, labels_train, column_labels_train, scoring='roc_auc_ovo')
    reduced_features, reduced_features_index, collection_coef_matrix = determine_important_features(features_train, labels_train, column_labels_train, return_collection_coef_matrix=True)

    #generate matrix
    morphology = df_train.morphology_clone.to_numpy()
    cell_names = df_train.cell_name.to_numpy()
    generate_matching_plot_2model(features_train,labels_train,labels_imaging_modality_train,path_to_data,column_labels_train,morphology,cell_names)
    generate_matching_plot_2model(features_train[:, reduced_features_index], labels_train, labels_imaging_modality_train, path_to_data, np.array(column_labels_train)[reduced_features_index], morphology, cell_names)
