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
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm
from hindbrain_structure_function.functional_type_prediction.FK_tools.branching_angle_calculator import *
import math

def calculate_all_dot_product(query_neuron,target_neuron):
    query_array = query_neuron.loc[:, ['x', 'y', 'z']].to_numpy()
    target_array = target_neuron.loc[:, ['x', 'y', 'z']].to_numpy()

    distances, indices = calculate_all_distances(query_neuron, target_neuron, return_indices=True)

    query_tangent_vectors = navis.Dotprops(query_array, k=5).vect
    target_tangent_vectors = navis.Dotprops(target_array, k=5).vect
    all_dotproducts = []
    for i in range(query_tangent_vectors.shape[0]):
        abs_dp = np.abs(np.dot(query_tangent_vectors[i], target_tangent_vectors[indices[i]]))
        all_dotproducts.append(abs_dp)
    return all_dotproducts

def calculate_all_distances(query_neuron,target_neuron,return_indices=False):
    query_array = query_neuron.loc[:,['x','y','z']].to_numpy()
    target_array = target_neuron.loc[:,['x','y','z']].to_numpy()
    tree = KDTree(target_array)

    #query the array

    d,i = tree.query(query_array)
    if return_indices:
        return d,i
    else:
        return d

def calculate_both(query_neuron,target_neuron):
    dp = calculate_all_dot_product(query_neuron,target_neuron)
    d = calculate_all_distances(query_neuron,target_neuron)
    both_combined = np.stack([d,dp],axis=1)
    return both_combined

def calculate_matrix_all_cells_in_df(df,swc_smooth=True,x_edges=None,y_edges=None,bins=15):
    all_tested = None



    #all cells
    for i1,cell1 in df.iterrows():
        if swc_smooth:
            try:
                query_neuron = df.loc[i1,'swc_smooth'].nodes
            except:
                query_neuron = df.loc[i1, 'swc'].nodes
        else:
            query_neuron = df.loc[i1, 'swc'].nodes
        for i2, cell2 in df.iterrows():
            target_neuron = df.loc[i2,'swc_smooth'].nodes
            result = calculate_both(query_neuron, target_neuron)
            if type(all_tested) == type(None):
                all_tested = result
            else:
                all_tested = np.concatenate([all_tested, result])
    if type(x_edges) == type(None):
        matrix, x_edges, y_edges =  np.histogram2d(all_tested[:,0],all_tested[:,1],bins=bins,density=True)
    else:
        matrix, x_edges, y_edges = np.histogram2d(all_tested[:, 0], all_tested[:, 1], bins=[x_edges,y_edges], density=True)
    return matrix, x_edges, y_edges

def plot_matrix(matrix,x_edges,y_edges,title='matrix distance vs dotproduct'):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix.T, origin='lower', aspect='auto',
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    plt.colorbar(im, ax=ax,location='right',pad=0.2)
    ax.set_xlabel('distance')
    ax.set_ylabel('absolute dot-product')
    #ax.axis('equal')
    # Draw bin edges
    # for x_edge in x_edges:
    #     ax.axvline(x=x_edge, color='r', linestyle='--', linewidth=0.5,alpha=0.3)
    # for y_edge in y_edges:
    #     ax.axhline(y=y_edge, color='r', linestyle='--', linewidth=0.5,alpha=0.3)

    # Annotate the edges with lower and upper bounds of each bin
    for i in range(len(x_edges) - 1):
        x_lower = x_edges[i]
        x_upper = x_edges[i + 1]
        x_center = (x_lower + x_upper) / 2
        ax.text(x_center, y_edges[0] - (y_edges[1] - y_edges[0]) * 0.2, f'{x_lower:.2f}-{x_upper:.2f}', ha='center', va='top', color='blue', fontsize=8, rotation=90)
        ax.text(x_center, y_edges[-1] + (y_edges[1] - y_edges[0]) * 0.2, f'{x_lower:.2f}-{x_upper:.2f}', ha='center', va='bottom', color='blue', fontsize=8, rotation=90)

    for j in range(len(y_edges) - 1):
        y_lower = y_edges[j]
        y_upper = y_edges[j + 1]
        y_center = (y_lower + y_upper) / 2
        ax.text(x_edges[0] - (x_edges[1] - x_edges[0]) * 0.2, y_center, f'{y_lower:.2f}-{y_upper:.2f}', ha='right', va='center', color='blue', fontsize=8)
        ax.text(x_edges[-1] + (x_edges[1] - x_edges[0]) * 0.2, y_center, f'{y_lower:.2f}-{y_upper:.2f}', ha='left', va='center', color='blue', fontsize=8)

    plt.subplots_adjust(left=0.2, right=0.8, top=0.7, bottom=0.2)  # Adjust margins
    plt.axis('off')
    plt.title(title, pad=70)


    plt.show()


if __name__ == "__main__":
    def symmetric_log_transform(x, linthresh=1):
        return np.sign(x) * np.log1p(np.abs(x / linthresh))
    name_time = datetime.now()
    # set path
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')   # Ensure this path is set in path_configuration.txt
    #load em data
    # all_cells_em = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["em"])
    # all_cells_em = all_cells_em.sort_values('classifier')
    # all_cells_em.loc[:, 'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_em['swc']]
    # all_cells_em.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_em['swc']]
    #load pa cells
    all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"],use_smooth=False)
    all_cells_pa_smooth = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"],use_smooth=True)
    all_cells_pa.loc[:,'swc_smooth'] = all_cells_pa_smooth['swc']
    all_cells_pa = all_cells_pa.dropna(subset='swc',axis=0)
    all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_pa['swc']]
    all_cells_pa.loc[:, 'swc_smooth'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_pa['swc_smooth']]
    all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_pa['swc']]
    all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_pa['swc_smooth']]

    #load clem cells
    # all_cells_clem = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["clem"],load_repaired=True)
    # all_cells_clem = all_cells_clem.dropna(subset='swc',axis=0)
    # all_cells_clem.loc[:, 'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_clem['swc']]
    # all_cells_clem.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_clem['swc']]



    #just two dts

    all_dt = all_cells_pa.loc[all_cells_pa['function']=='dynamic_threshold',:]
    query_neuron = all_dt.iloc[0]['swc'].nodes
    target_neuron = all_dt.iloc[2]['swc'].nodes

    aaa = calculate_both(query_neuron,target_neuron)


    matrix, x_edges, y_edges =  np.histogram2d(aaa[:,0],aaa[:,1],bins=10,density=True)
    # Plot the 2D histogram

    plot_matrix(matrix,x_edges,y_edges,'2 DTs')



    #for all dts


    matrix_all, x_edges, y_edges =  calculate_matrix_all_cells_in_df(all_cells_pa)

    all_dt = all_cells_pa.loc[all_cells_pa['function'] == 'dynamic_threshold', :]
    matrix_dt, x_edges, y_edges =  calculate_matrix_all_cells_in_df(all_dt)

    plot_matrix(matrix_dt,x_edges,y_edges,'All DTs')


    #combine
    temp_array = matrix_dt/matrix_all
    log_array = np.empty_like(matrix_dt)
    for i1 in range(matrix_dt.shape[0]):
        for i2 in range(matrix_dt.shape[1]):
            try:
                log_array[i1,i2] = math.log2(temp_array[i1,i2])
            except:
                log_array[i1, i2] = np.nan
    plt.imshow(log_array)

















