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
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import  *
from matplotlib import colors
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette

import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import seaborn as sns
from copy import deepcopy

if __name__ == "__main__":
    def symmetric_log_transform(x, linthresh=1):
        return np.sign(x) * np.log1p(np.abs(x / linthresh))
    name_time = datetime.now()
    # Set the base path for data by reading from a configuration file; ensures correct data location is used.
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')   # Ensure this path is set in path_configuration.txt


    all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"])


    #extract tags

    cell_type_categories = {'morphology': ['ipsilateral', 'contralateral'],
                            'neurotransmitter': ['inhibitory', 'excitatory'],
                            'function': ['integrator', 'dynamic_threshold', 'dynamic threshold', 'motor_command', 'motor command']}
    for i, cell in all_cells_pa.iterrows():
        if type(cell.cell_type_labels) == list:
            for label in cell.cell_type_labels:
                if label in cell_type_categories['morphology']:
                    all_cells_pa.loc[i, 'morphology'] = label
                elif label in cell_type_categories['function']:
                    all_cells_pa.loc[i, 'function'] = label
                elif label in cell_type_categories['neurotransmitter']:
                    all_cells_pa.loc[i, 'neurotransmitter'] = label
        all_cells_pa.loc[i, 'swc'].name = all_cells_pa.loc[i, 'swc'].name + " NT: " + all_cells_pa.loc[i, 'neurotransmitter']

    #sort dfs
    all_cells_pa = all_cells_pa.sort_values(['function','morphology','neurotransmitter'])


    #shift cells
    for i, cell in all_cells_pa.iterrows():
        x_shift = cell['swc'].nodes.loc[0,'x']
        y_shift = cell['swc'].nodes.loc[0, 'y']
        z_shift = cell['swc'].nodes.loc[0, 'z']
        all_cells_pa.loc[i,'swc'].nodes.loc[:, 'x'] = cell['swc'].nodes.loc[:, 'x'] - x_shift
        all_cells_pa.loc[i, 'swc'].nodes.loc[:, 'y'] = cell['swc'].nodes.loc[:, 'y'] - y_shift
        all_cells_pa.loc[i, 'swc'].nodes.loc[:, 'z'] = cell['swc'].nodes.loc[:, 'z'] - z_shift


    nb_df = nblast_two_groups(all_cells_pa, all_cells_pa)



    df_nb = nb_df

    do_best_em_match4pa = True
    do_best_pa_match4em = True
    mark_in_manual = True
    do_SymLogNorm = False
    figsize = (20, 20)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    if do_SymLogNorm:
        ax[0].pcolormesh(nb_df, norm=colors.SymLogNorm(linthresh=1, vmin=np.min(symmetric_log_transform(df_nb)), vmax=np.max(symmetric_log_transform(df_nb))))
    else:
        cbar = ax[0].pcolormesh(df_nb,vmin=-0.5,vmax=0.5)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].axis('off')



    cell_type_categories = {'morphology': ['ipsilateral', 'contralateral'],
                            'neurotransmitter': ['inhibitory', 'excitatory'],
                            'function': ['motor_command','integrator', 'dynamic_threshold']}#'motor command','dynamic threshold'
    color_cell_type_dict = {
        "integrator_ipsilateral": '#feb326b3',
        "integrator_contralateral": '#e84d8ab3',
        "motor_command_ipsilateral": '#7f58afb3',
        "motor command_ipsilateral": '#7f58afb3',
        "motor_command_contralateral": '#7f58afb3',
        "motor command_contralateral": '#7f58afb3',

        "dynamic_threshold_ipsilateral": '#64c5ebb3',
        "dynamic threshold_ipsilateral": '#64c5ebb3',
        "dynamic threshold_contralateral": '#64c5ebb3',
        "dynamic_threshold_contralateral": '#64c5ebb3',
    }

    label_df_pa = all_cells_pa.reset_index()
    label_df_pa['double_label'] = label_df_pa.apply(lambda x: x['function'] +"_"+ x['morphology'], axis=1)

    for dl in color_cell_type_dict.keys():



        color = color_cell_type_dict[dl]

        label_df_pa.loc[label_df_pa['double_label']==dl,:]
        min = np.min(label_df_pa.loc[label_df_pa['double_label']==dl,:].index)
        max = np.max(label_df_pa.loc[label_df_pa['double_label'] == dl, :].index)
        ax[0].plot([min, max+1], [df_nb.shape[0]+1, df_nb.shape[0]+1],c=color,lw=8,solid_capstyle='butt')
        ax[1].plot([min, max + 1], [df_nb.shape[0] + 1, df_nb.shape[0] + 1], c=color,lw=8,solid_capstyle='butt')

    for dl in color_cell_type_dict.keys():



        color = color_cell_type_dict[dl]

        label_df_pa.loc[label_df_pa['double_label']==dl,:]
        min = np.min(label_df_pa.loc[label_df_pa['double_label']==dl,:].index)
        max = np.max(label_df_pa.loc[label_df_pa['double_label'] == dl, :].index)
        ax[0].plot([-1, -1],[min, max+1],c=color,lw=8,solid_capstyle='butt')





    fig.colorbar(cbar)
    plt.show()



    #hirarchical clustering
    nb_mean = (nb_df + nb_df.T) / 2
    nb_dist = 1 - nb_mean

    from scipy.cluster.hierarchy import fcluster

    # To generate a linkage, we have to bring the matrix from square-form to vector-form
    aba_vec = squareform(nb_dist, checks=False)

    # Generate linkage
    Z = linkage(aba_vec, method='ward')
    #clustering
    clusters_by_distance = fcluster(Z, 1, criterion='distance')
    clusters_by_number = fcluster(Z, 10, criterion='maxclust')

    clusters = clusters_by_number
    no_of_ax = int(np.ceil(np.sqrt(len(np.unique(clusters)))))
    fig,ax = plt.subplots(no_of_ax,no_of_ax,figsize=(30,30))
    x_ax = 0
    y_ax = 0

    for cluster_no in np.unique(clusters):
        cells2plot_index = np.where(clusters==cluster_no)
        temp_df = all_cells_pa.iloc[list(cells2plot_index[0]),:]
        navis.plot2d(list(temp_df['swc']),  alpha=1, linewidth=0.5,
                     method='2d', view='xy', group_neurons=True,ax=ax[x_ax][y_ax])
        ax[x_ax][y_ax].set_aspect('equal')
        ax[x_ax][y_ax].axis('off')
        x_ax+=1

        if x_ax == no_of_ax:
            x_ax = 0
            y_ax+=1

    plt.show()
    all_cells_pa['c'] = clusters
    all_cells_pa.groupby('c')['function'].value_counts(normalize=False).unstack().fillna(0)