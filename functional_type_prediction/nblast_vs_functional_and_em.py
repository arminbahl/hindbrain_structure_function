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
import navis
from navis.nbl.smat import Lookup2d, LookupDistDotBuilder, Digitizer
import h5py
from tqdm import tqdm
from hindbrain_structure_function.functional_type_prediction.nblast_matrix_navis import *


def plot_nblast_matrix(nb_df,
                       df1,
                       df2,
                       do_best_em_match4pa=True,
                       do_best_pa_match4em=True,
                       do_SymLogNorm=False,
                       figsize=(20, 20),
                       v_limit=0.4,
                       normalize=False,
                       use_gregor_classifier=False, title='nblast matrix'):
    if normalize:
        nb_df = nb_df / np.max(nb_df)

    # init figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    # plot
    if type(v_limit) == bool:
        if do_SymLogNorm:
            ax.pcolormesh(nb_df, norm=colors.SymLogNorm(linthresh=1, vmin=np.min(symmetric_log_transform(nb_df)), vmax=np.max(symmetric_log_transform(nb_df))))
        else:
            cbar = ax.pcolormesh(nb_df)


    else:
        if normalize == False:
            cbar = ax.pcolormesh(nb_df, vmin=-abs(v_limit), vmax=abs(v_limit))
        else:
            raise TypeError("No normalization while v_limit is set")

    # adjust axis
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

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

    # set x axis labeling
    df2 = df2.reset_index()
    df2['double_label'] = df2.apply(lambda x: x['function'] + "_" + x['morphology'], axis=1)
    for dl in color_cell_type_dict.keys():
        color = color_cell_type_dict[dl]

        min = np.min(df2.loc[df2['double_label'] == dl, :].index)
        max = np.max(df2.loc[df2['double_label'] == dl, :].index)
        ax.plot([min, max + 1], [nb_df.shape[0] + 1, nb_df.shape[0] + 1], c=color, lw=8, solid_capstyle='butt')

    df1 = df1.reset_index()

    ax.plot([-1, -1], [0, nb_df.shape[0]], c='gray', lw=8, solid_capstyle='butt', alpha=0.5)
    if use_gregor_classifier:

        for dl in np.unique(df1.classifier):

            min = np.min(df1.loc[df1['classifier'] == dl, :].index)
            max = np.max(df1.loc[df1['classifier'] == dl, :].index) + 1

            if min != 0:
                ax.plot([-1.5, -0.5], [min, min], c='k', lw=2, solid_capstyle='butt', alpha=1)
            if max != nb_df.shape[0]:
                ax.plot([-1.5, -0.5], [max, max], c='k', lw=2, solid_capstyle='butt', alpha=1)
            ax.text(-2, np.mean([min, max]), dl, horizontalalignment='right')
    else:
        try:
            for dl in np.unique(df1.nblast_classifier):

                min = np.min(df1.loc[df1['nblast_classifier'] == dl, :].index)
                max = np.max(df1.loc[df1['nblast_classifier'] == dl, :].index) + 1

                if min != 0:
                    ax.plot([-1.5, -0.5], [min, min], c='k', lw=2, solid_capstyle='butt', alpha=1)
                if max != nb_df.shape[0]:
                    ax.plot([-1.5, -0.5], [max, max], c='k', lw=2, solid_capstyle='butt', alpha=1)
                ax.text(-2, np.mean([min, max]), dl, horizontalalignment='right')
        except:
            pass

    fig.colorbar(cbar)
    plt.suptitle(title)
    plt.show()


if __name__ == "__main__":
    def symmetric_log_transform(x, linthresh=1):
        return np.sign(x) * np.log1p(np.abs(x / linthresh))
    name_time = datetime.now()
    # set path
    path_to_data = get_base_path()  # Ensure this path is set in path_configuration.txt
    #load em data
    all_cells_em = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["em"])
    all_cells_em = all_cells_em.sort_values('classifier')

    #load pa cells
    all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"],use_smooth=False)
    all_cells_pa_smooth = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"],use_smooth=True)
    all_cells_pa.loc[:,'swc_smooth'] = all_cells_pa_smooth['swc']
    all_cells_pa = all_cells_pa.dropna(subset='swc',axis=0)

    #load clem cells
    all_cells_clem = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["clem"],load_repaired=True)
    all_cells_clem = all_cells_clem.dropna(subset='swc',axis=0)


    #prune
    all_cells_em.loc[:,'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_em['swc']]
    all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_pa['swc']]
    all_cells_pa.loc[:, 'swc_smooth'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_pa['swc_smooth']]
    all_cells_clem.loc[:, 'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_clem['swc']]


    all_cells_em.loc[:,'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_em['swc']]
    all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_pa['swc']]
    all_cells_pa.loc[:, 'swc_smooth'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_pa['swc_smooth']]
    all_cells_clem.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_clem['swc']]

    #stack
    all_cells = pd.concat([all_cells_clem,all_cells_pa,all_cells_em], axis=0)




    #nblast
    smat_fish = load_zebrafish_nblast_matrix(return_smat_obj=True,prune=False,modalities=["pa",'clem'])
    all_cells_pa = all_cells_pa.sort_values(['function','morphology'])
    nb_em_pa = nblast_two_groups_custom_matrix(all_cells_pa, all_cells_pa,custom_matrix=smat_fish)
    plot_nblast_matrix(nb_em_pa, all_cells_em, all_cells_pa, title='nblast all')


    #plot



    fig, ax = plt.subplots(nrows=1, ncols=1)
    # plot

    df2 = all_cells_pa
    df2 = df2.reset_index(drop=True)

    cbar = ax.pcolormesh(nb_em_pa,vmin=0,vmax=nb_em_pa[nb_em_pa<0.95].max().max())

    # adjust axis
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

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

    # set x axis labeling

    df2['double_label'] = df2.apply(lambda x: x['function'] + "_" + x['morphology'], axis=1)
    for dl in color_cell_type_dict.keys():
        color = color_cell_type_dict[dl]

        min = np.min(df2.loc[df2['double_label'] == dl, :].index)
        max = np.max(df2.loc[df2['double_label'] == dl, :].index)
        ax.plot([min, max + 1], [nb_em_pa.shape[0] + 1, nb_em_pa.shape[0] + 1], c=color, lw=8, solid_capstyle='butt')
        ax.plot([-1, -1], [min, max+1], c=color, lw=8, solid_capstyle='butt')








    fig.colorbar(cbar)
    plt.suptitle('pa vs pa')
    plt.show()

