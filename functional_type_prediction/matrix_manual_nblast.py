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
    path_to_data = get_base_path()  # Ensure this path is set in path_configuration.txt

    all_cells_em = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["em"])
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
    all_cells_em = all_cells_em.sort_values('classifier')

    #all cell shenanigans
    
    all_cells_em_shifted = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["em"])
    all_cells_pa_shifted = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"])
    
    cell_type_categories = {'morphology': ['ipsilateral', 'contralateral'],
                            'neurotransmitter': ['inhibitory', 'excitatory'],
                            'function': ['integrator', 'dynamic_threshold', 'dynamic threshold', 'motor_command', 'motor command']}
    for i, cell in all_cells_pa_shifted.iterrows():
        if type(cell.cell_type_labels) == list:
            for label in cell.cell_type_labels:
                if label in cell_type_categories['morphology']:
                    all_cells_pa_shifted.loc[i, 'morphology'] = label
                elif label in cell_type_categories['function']:
                    all_cells_pa_shifted.loc[i, 'function'] = label
                elif label in cell_type_categories['neurotransmitter']:
                    all_cells_pa_shifted.loc[i, 'neurotransmitter'] = label
        all_cells_pa_shifted.loc[i, 'swc'].name = all_cells_pa_shifted.loc[i, 'swc'].name + " NT: " + all_cells_pa_shifted.loc[i, 'neurotransmitter']

    #sort dfs
    all_cells_pa_shifted = all_cells_pa_shifted.sort_values(['function','morphology','neurotransmitter'])
    all_cells_em_shifted = all_cells_em_shifted.sort_values('classifier')
    
    all_cells_shifted = pd.concat([all_cells_pa_shifted,all_cells_em_shifted])
    all_cells_shifted = all_cells_shifted.reset_index(drop=True)

    for i, cell in all_cells_shifted.iterrows():
        x_shift = cell['swc'].nodes.loc[0,'x']
        y_shift = cell['swc'].nodes.loc[0, 'y']
        z_shift = cell['swc'].nodes.loc[0, 'z']
        all_cells_shifted.loc[i,'swc'].nodes.loc[:, 'x'] = cell['swc'].nodes.loc[:, 'x'] - x_shift
        all_cells_shifted.loc[i, 'swc'].nodes.loc[:, 'y'] = cell['swc'].nodes.loc[:, 'y'] - y_shift
        all_cells_shifted.loc[i, 'swc'].nodes.loc[:, 'z'] = cell['swc'].nodes.loc[:, 'z'] - z_shift



    manual_df = pd.read_excel(path_to_data / "em_zfish1" / 'manual-cell-assignment_em2paGFP.xlsx')
    manual_df = manual_df.iloc[:, :6]

    #new matrix manual

    manual_matrix = pd.DataFrame(data=0,columns=all_cells_pa.cell_name,index=all_cells_em.cell_name)

    for i,cell in manual_df.iterrows():
        temp_match_cells = cell['assigned paGFP cell(s)']
        temp_match_cells = temp_match_cells.lstrip().rstrip()
        if temp_match_cells[-1] == ',':
            temp_match_cells = temp_match_cells[:-1]

        temp_match_cells = [x.lstrip().rstrip() for x in temp_match_cells.split(',')]
        for matched_cell in temp_match_cells:
            try:
                manual_matrix.loc[str(cell['EM cell']), matched_cell] = 1
            except:
                manual_matrix.loc[cell['EM cell'], matched_cell] = 1

    manual_matrix = manual_matrix.dropna(axis=1)
    #nblast
    nb_df = nblast_two_groups(all_cells_em, all_cells_em)
    nb_compare = nblast_two_groups(all_cells_em, all_cells_pa)
    nb_compare_shifted = nblast_two_groups(all_cells_shifted.loc[all_cells_shifted['imaging_modality'] == 'EM', :], all_cells_shifted.loc[all_cells_shifted['imaging_modality'] != 'EM', :])

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
        temp_df = all_cells_em.iloc[list(cells2plot_index[0]),:]
        navis.plot2d(list(temp_df['swc']),  alpha=1, linewidth=0.5,
                     method='2d', view='xy', group_neurons=True,ax=ax[x_ax][y_ax])
        ax[x_ax][y_ax].set_aspect('equal')
        ax[x_ax][y_ax].axis('off')
        x_ax+=1

        if x_ax == no_of_ax:
            x_ax = 0
            y_ax+=1

    plt.show()
    #plotting
    os.makedirs(path_to_data / "make_figures_FK_output" / 'compare_nblast2manual_assignment',exist_ok=True)
    save_path_fig = path_to_data / "make_figures_FK_output" / 'compare_nblast2manual_assignment'

    figsize = (20,20)




    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(manual_matrix)
    ax.set_aspect('equal', adjustable='box')
    plt.suptitle("Manual cell assignment matrix")
    plt.savefig(save_path_fig / 'Manual cell assignment matrix.pdf')
    plt.savefig(save_path_fig / 'Manual cell assignment matrix.png')
    plt.show()

    def plot_matrices(df_nb,df_manual,do_best_em_match4pa=True,do_best_pa_match4em=True,mark_in_manual=True,do_SymLogNorm=False,figsize = (20,20)):


        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        if do_SymLogNorm:
            ax[0].pcolormesh(df_nb, norm=colors.SymLogNorm(linthresh=1, vmin=np.min(symmetric_log_transform(df_nb)), vmax=np.max(symmetric_log_transform(df_nb))),alpha=df_nb>0)
        else:
            ax[0].pcolormesh(df_nb,alpha=df_nb>0)
        ax[0].set_aspect('equal', adjustable='box')
        ax[1].pcolormesh(df_manual)
        ax[1].set_aspect('equal', adjustable='box')
        ax[0].axis('off')
        ax[1].axis('off')
        if do_best_pa_match4em:
            for i, em_cell in df_nb.iterrows():
                x = em_cell.index.get_loc(em_cell.idxmax())
                y = df_nb.index.get_loc(i)
                my_sq = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]])
                ax[0].plot(my_sq[:, 0], my_sq[:, 1], color='red')
                if mark_in_manual:
                    ax[1].plot(my_sq[:, 0], my_sq[:, 1], color='red')
        if do_best_em_match4pa:
            for i, pa_cell in df_nb.T.iterrows():
                y = pa_cell.index.get_loc(pa_cell.idxmax())
                x = df_nb.columns.get_loc(i)
                my_sq = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]])
                ax[0].plot(my_sq[:, 0], my_sq[:, 1], color='white')
                if mark_in_manual:
                    ax[1].plot(my_sq[:, 0], my_sq[:, 1], color='white')



    plot_matrices(nb_compare,manual_matrix)
    plt.suptitle("Nblast cell assignment matrix")
    plt.savefig(save_path_fig / 'Nblast cell assignment matrix.pdf')
    plt.savefig(save_path_fig / 'Nblast cell assignment matrix.png')
    plt.show()

    plot_matrices(nb_compare, manual_matrix,do_SymLogNorm=True)
    plt.suptitle("Nblast cell assignment matrix\nSymLogNorm")
    plt.savefig(save_path_fig / 'Nblast cell assignment matrix SymLogNorm.pdf')
    plt.savefig(save_path_fig / 'Nblast cell assignment matrix SymLogNorm.png')
    plt.show()


    plot_matrices(nb_compare_shifted,manual_matrix)
    plt.suptitle("Nblast shifted cell assignment matrix")
    plt.savefig(save_path_fig / 'Nblast shifted cell assignment matrix.pdf')
    plt.savefig(save_path_fig / 'Nblast shifted cell assignment matrix.png')
    plt.show()

    plot_matrices(nb_compare_shifted, manual_matrix, do_SymLogNorm=True)
    plt.suptitle("Nblast shifted cell assignment matrix\nSymLogNorm")
    plt.savefig(save_path_fig / 'Nblast shifted cell assignment matrix SymLogNorm.pdf')
    plt.savefig(save_path_fig / 'Nblast shifted cell assignment matrix SymLogNorm.png')
    plt.show()

    avg_nb = (nb_compare+nb_compare_shifted)/2

    plot_matrices(avg_nb, manual_matrix, do_SymLogNorm=False)
    plt.suptitle("Nblast shifted and normal mean\n cell assignment matrix")
    plt.savefig(save_path_fig / 'Nblast shifted and normal mean cell assignment matrix.pdf')
    plt.savefig(save_path_fig / 'Nblast shifted and normal mean cell assignment matrix.png')
    plt.show()


    plot_matrices(avg_nb, manual_matrix, do_SymLogNorm=True)
    plt.suptitle("Nblast shifted and normal mean\n cell assignment matrix\nSymLogNorm")
    plt.savefig(save_path_fig / 'Nblast shifted and normal mean cell assignment matrix SymLogNorm.pdf')
    plt.savefig(save_path_fig / 'Nblast shifted and normal mean cell assignment matrix SymLogNorm.png')
    plt.show()




    correlation = []
    for i, em_cell in manual_matrix.iterrows():
        correlation.append(np.corrcoef(em_cell,avg_nb.loc[i,:])[0,1])
        if np.isnan(correlation[-1]):
            correlation[-1]






    #test area
    df_nb = avg_nb
    df_manual= manual_matrix
    do_best_em_match4pa = True
    do_best_pa_match4em = True
    mark_in_manual = True
    do_SymLogNorm = False
    figsize = (20, 20)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    if do_SymLogNorm:
        ax[0].pcolormesh(df_nb, norm=colors.SymLogNorm(linthresh=1, vmin=np.min(symmetric_log_transform(df_nb)), vmax=np.max(symmetric_log_transform(df_nb))))
    else:
        ax[0].pcolormesh(df_nb)
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].pcolormesh(df_manual)
    ax[1].set_aspect('equal', adjustable='box')
    ax[0].axis('off')
    ax[1].axis('off')
    if do_best_pa_match4em:
        for i, em_cell in df_nb.iterrows():
            x = em_cell.index.get_loc(em_cell.idxmax())
            y = df_nb.index.get_loc(i)
            my_sq = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]])
            ax[0].plot(my_sq[:, 0], my_sq[:, 1], color='red')
            if mark_in_manual:
                ax[1].plot(my_sq[:, 0], my_sq[:, 1], color='red')
    if do_best_em_match4pa:
        for i, pa_cell in df_nb.T.iterrows():
            y = pa_cell.index.get_loc(pa_cell.idxmax())
            x = avg_nb.columns.get_loc(i)
            my_sq = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1], [x, y]])
            ax[0].plot(my_sq[:, 0], my_sq[:, 1], color='white')
            if mark_in_manual:
                ax[1].plot(my_sq[:, 0], my_sq[:, 1], color='white')

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

    label_df_em = all_cells_em.reset_index()

    em_colors = {'Class 1, ipsilateral integrator':"#9e0142",
                 'Class 1, putative motor command':"#5e4fa2",
                 'Class 2, contralateral integrator':'#d53e4f',
                 'Class 4, Raphe neuron':'#3288bd',
                 'Class 5 - ipsilateral posterior axon':'#f46d43',
                 'Class 5, contralateral motor':'#66c2a5',
                 'Class 6 - contralateral posterior axon':"#fdae61",
                 'Class 7, dynamic threshold':'#abdda4',
                 'cerebellum-projecting':'#fee08b',
                 'contralateral axon':'#e6f598',
                 'ipsilateral axon':'#ffffbf'}
    from matplotlib.patches import Patch
    legend_em = []


    for cell_type in np.unique(label_df_em.classifier):
        color = em_colors[cell_type]
        label_df_em.loc[all_cells_em['classifier'] == cell_type, :]
        min = np.min(label_df_em.loc[label_df_em['classifier'] == cell_type, :].index)
        max = np.max(label_df_em.loc[label_df_em['classifier'] == cell_type, :].index)
        ax[0].plot([-1, -1], [min, max + 1], lw=8, solid_capstyle='butt',c=color)
        ax[1].plot([-1, -1], [min, max + 1], lw=8, solid_capstyle='butt',c=color)
        legend_em.append( Patch(facecolor=color, edgecolor=color,
                         label=cell_type))
        ax[0].text(-20,np.mean([max,min]),cell_type)
        print(np.mean([max,min]))





    plt.show()


    #nblast against itself
