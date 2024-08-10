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
from hindbrain_structure_function.visualization.make_figures_FK import *
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
import winsound

def extract_prediction(result):
    cleaned = [x.replace('"', "") for x in result.split(' ')[2:]]
    a_proba = np.array([eval(x[3:]) for x in cleaned])
    a_label = np.array([x[:2] for x in cleaned])
    prediced_label = a_label[np.argmax(a_proba)]

    return prediced_label
if __name__ == '__main__':

    #set variables
    np.set_printoptions(suppress=True)
    path_to_data = get_base_path()
    brain_meshes = load_brs(path_to_data, 'raphe')
    width_brain = 495.56
    path_to_figure_dir = path_to_data / 'make_figures_FK_output' / 'EmJ_connectome_enhancement'
    os.makedirs(path_to_figure_dir,exist_ok=True)

    target_cell_id = 'cell_576460752681311362'

    #load cells

    cells_clem1 = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem'], load_repaired=True,load_both=False,mirror=False)
    cells_clem2 = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['prediction_project'],load_repaired=True,load_both=False)
    cells_clem = pd.concat([cells_clem1,cells_clem2])
    cells_clem = cells_clem.reset_index(drop=True)
    cells_clem = cells_clem.loc[cells_clem['type'] != 'axon', :]
    cells_clem = cells_clem.drop_duplicates(subset='cell_name')



    for i,cell in cells_clem.iterrows():
        try:
            temp_meta_path = cell['metadata_path']
            temp_meta_path = Path(str(temp_meta_path)[:-4] + '_with_prediction.txt')

            with open(temp_meta_path, 'r') as f:
                t = f.read()

            meta_data_list = [x for x in t.split('\n') if x != ""]

            cells_clem.loc[i,'prediction'] = extract_prediction(meta_data_list[15])
            cells_clem.loc[i, 'prediction_reduced'] = extract_prediction(meta_data_list[16])
            cells_clem.loc[i, 'nblast_test'] = eval(meta_data_list[17].split(' ')[2])
            cells_clem.loc[i, 'nblast_test_specific'] = eval(meta_data_list[18].split(' ')[2])
            cells_clem.loc[i, 'proba_test'] = eval(meta_data_list[19].split(' ')[2])
            cells_clem.loc[i, 'proba_test_reduced'] = eval(meta_data_list[20].split(' ')[2])
            cells_clem.loc[i, 'nov_OCSVM'] = eval(meta_data_list[21].split(' ')[2])
            cells_clem.loc[i, 'nov_OCSVM_reduced'] = eval(meta_data_list[22].split(' ')[2])
            cells_clem.loc[i, 'nov_IF'] = eval(meta_data_list[23].split(' ')[2])
            cells_clem.loc[i, 'nov_IF_reduced'] = eval(meta_data_list[24].split(' ')[2])
            cells_clem.loc[i, 'nov_LOF'] = eval(meta_data_list[25].split(' ')[2])
            cells_clem.loc[i, 'nov_LOF_reduced'] = eval(meta_data_list[26].split(' ')[2])
        except:
            pass




    color_dict = {
        "II": '#feb326b3',
        "CI": '#e84d8ab3',
        "DT": '#64c5ebb3',
        "MC": '#7f58afb3',
    }

    acronym_convert = {
        "ipsilateral integrator":"II",
        "contralateral integrator":"CI",
        "dynamic threshold":"DT",
        "motor command":"MC",
    }

    color_dict_rgba = {
        "II": [254,179,38,1],
        "CI": [232,77,138,1],
        "DT": [100,197,235,1],
        "MC": [127,88,175,1],
    }



    #find postsynaptic to target cell

    load_synapse_clem(cells_clem.loc[cells_clem['cell_name']==target_cell_id,:], path_to_data, mapped=False)
    for i,cell in cells_clem.iterrows():
        temp_cell = load_synapse_clem(cell,path_to_data,mapped=False)
        cells_clem.loc[i,:] == temp_cell


    for it in ['presynaptic_cells','postsynaptic_cells']:
        cells_clem[it] = None
        cells_clem[it] = cells_clem[it].astype(object)


    #write synapse files
    for i,cell in cells_clem.iterrows():
        if cell.swc.connectors is not None:
            target_synapses = cell.swc.connectors
            target_post = target_synapses.loc[(target_synapses['type'] == 'post'), :]
            target_pre = target_synapses.loc[(target_synapses['type'] == 'pre'), :]
            for i2,cell2 in cells_clem.iterrows():
                if cell2.cell_name == cell.cell_name:
                    pass
                else:
                    if cell2['cell_name'] == 'cell_576460752789011603' and cell['cell_name'] == target_cell_id:
                        pass
                    if cell2['dendrites_id'] in list(target_pre.connector_id):
                        if cells_clem.loc[i, 'postsynaptic_cells'] is None:
                            cells_clem.loc[i, 'postsynaptic_cells'] = [cell2.cell_name]
                        else:
                            cells_clem.loc[i, 'postsynaptic_cells'].append(cell2.cell_name)

                    if cell2['axon_id'] in list(target_post.connector_id):
                        if cells_clem.loc[i, 'presynaptic_cells'] is None:
                            cells_clem.loc[i, 'presynaptic_cells'] = [cell2.cell_name]
                        else:
                            cells_clem.loc[i, 'presynaptic_cells'].append(cell2.cell_name)







    #postsynaptic2cell_576460752681311362

    postsynaptic2cell_576460752681311362 = ['576460752307754417','576460752653085234','576460752308941605','576460752729891924','576460752789011603',
                                            '576460752619742263','576460752663151851','576460752692925619','576460752702641204','576460752671285639',
                                            '576460752649503674','576460752703267321','576460752385724722','576460752677927306','576460752683423174',
                                            '576460752610682677','576460752616132567','576460752665808837','576460752330776649','576460752658326542',
                                            '576460752710312114','576460752651541215','576460752698460747','576460752677174876','576460752685678166',
                                            '576460752728175444','576460752634468945']
    postsynaptic2cell_576460752681311362 = ['cell_'+x for x in postsynaptic2cell_576460752681311362]

    enhanced_connectome = cells_clem.loc[(cells_clem['cell_name'].isin(postsynaptic2cell_576460752681311362))|(cells_clem['cell_name']==target_cell_id),:]
    for i,cell in enhanced_connectome.iterrows():
        if cell['function'] != 'nan' and cell['function'] is not np.nan:
            enhanced_connectome.loc[i,'cell_class'] = cell['function']
            if cell['function'] == 'integrator':
                enhanced_connectome.loc[i, 'cell_class'] = cell['morphology'] +" "+ cell['function']
            enhanced_connectome.loc[i, 'cell_class'] = acronym_convert[enhanced_connectome.loc[i, 'cell_class']]
        else:
            enhanced_connectome.loc[i, 'cell_class'] = cell['prediction']

    color_list = [color_dict[x] for x in enhanced_connectome['cell_class']]

    #plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))

    color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)
    projection = 'z'
    if projection == "z":
        view = ('x', "-y")  # Set the 2D view to the X-Y plane for Z projection.
        ylim = [-850, -50]  # Define the Y-axis limits for the Z projection.
    elif projection == 'y':
        view = ('x', "z")  # Set the 2D view to the X-Z plane for Y projection.
        ylim = [-30, 300]
    ylim = [-850, -50]
    ylim = [-30, 300]

    navis.plot2d(brain_meshes, color=color_meshes,
                 alpha=0.2, linewidth=0.5, method='2d', view=view, group_neurons=False,
                 rasterize=True, ax=ax[0],
                 scalebar="20 um")

    navis.plot2d(navis.NeuronList(enhanced_connectome['swc']), color=color_list, alpha=1, linewidth=0.5,
                 method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[0])

    # navis.plot2d(navis.NeuronList(enhanced_connectome['soma_mesh']), color=color_list, alpha=1, linewidth=0.5,
    #              method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[0])

    # navis.plot2d(axon_mesh_list, color=color_list, alpha=1, linewidth=0.5,
    ax[0].set_aspect('equal')
    ax[0].axvline(250, color=(0.85, 0.85, 0.85, 0.2), linestyle='--', alpha=0.5, zorder=0)
    ax[0].set_xlim(50, 350)  # Standardize the plot dimensions.
    # Set specific limits for the Y-axis based on the projection.
    ax[0].set_facecolor('white')
    ax[0].axis('off')
    # f'{cell_class}\nCLEM: {row.iloc[0]}\nEM: {row.iloc[1]}\npaGFP: {row.iloc[2]}.pdf'

    # fig.savefig(path_to_figure_dir / f'{cell_class}_clem{row.iloc[0]}_em{row.iloc[1]}pa_{row.iloc[2]}.pdf', dpi=600)
    # fig = plt.figure(figsize=(12, 12))

    color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)
    projection = 'y'
    if projection == "z":
        view = ('x', "-y")  # Set the 2D view to the X-Y plane for Z projection.
        ylim = [-850, -50]  # Define the Y-axis limits for the Z projection.
    elif projection == 'y':
        view = ('x', "z")  # Set the 2D view to the X-Z plane for Y projection.
        ylim = [-80, 400]

    navis.plot2d(brain_meshes, color=color_meshes, volume_outlines=False,
                 alpha=0.2, linewidth=0.5, method='2d', view=view, group_neurons=False,
                 rasterize=True, ax=ax[1],
                 scalebar="20 um")

    navis.plot2d(navis.NeuronList(enhanced_connectome['swc']), color=color_list, alpha=1, linewidth=0.5,
                 method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[1])

    navis.plot2d(navis.NeuronList(enhanced_connectome['soma_mesh']), color=color_list, alpha=1, linewidth=0.5,
                 method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[1])

    ax[1].set_aspect('equal')
    ax[1].axvline(250, color=(0.85, 0.85, 0.85, 0.2), linestyle='--', alpha=0.5, zorder=0)
    ax[1].set_xlim(50, 350)  # Standardize the plot dimensions.
    # Set specific limits for the Y-axis based on the projection.
    ax[1].set_facecolor('white')
    ax[1].axis('off')
    # {cell_class}\nCLEM: {row.iloc[0]}\nEM: {row.iloc[1]}\npaGFP: {row.iloc[2]}
    fig.savefig(path_to_figure_dir / f'enhanced_connectome_{target_cell_id}.png', dpi=300)
    fig.savefig(path_to_figure_dir / f'enhanced_connectome_{target_cell_id}.pdf', dpi=300)

    import fnmatch



