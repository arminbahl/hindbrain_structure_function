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
    # set path
    path_to_data = get_base_path()  # Ensure this path is set in path_configuration.txt
    #load em data
    all_cells_em = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["em"])
    all_cells_em = all_cells_em.sort_values('classifier')
    #load pa cells
    all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"])
    all_cells_pa = all_cells_pa.dropna(subset='swc',axis=0)


    #categorize em cells contralateral
    width_brain = 495.56
    all_cells_em['contralateral'] =False
    for i,cell in all_cells_em.iterrows():
        if True in list(cell['swc'].nodes.loc[:,"x"]>(width_brain/2)):
            all_cells_em.loc[i,'contralateral'] = True



    #extract features from pa cell labels
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
    prune = True
    if prune:
        all_cells_em.loc[:,'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_em['swc']]
        all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_pa['swc']]

    sort_nblast=True
    if sort_nblast:

        # sort_by_nblast
        print('Start nblast')
        sort_nb = nblast_two_groups(all_cells_em, all_cells_em)
        work_nb = (sort_nb + sort_nb.T) / 2
        # nb_dist = 1 - work_nb
        # work_nb = squareform(work_nb, checks=False)
        # Z = linkage(work_nb, method='ward')

        sort_clusters = navis.nbl.make_clusters(work_nb, 1, criterion='height')
        all_cells_em['nblast_classifier'] = sort_clusters

        no_of_ax = int(np.ceil(np.sqrt(len(np.unique(sort_clusters)))))
        fig, ax = plt.subplots(no_of_ax, no_of_ax, figsize=(30, 30))
        x_ax = 0
        y_ax = 0

        for cluster_no in np.unique(sort_clusters):

            navis.plot2d(list(all_cells_em.loc[all_cells_em['nblast_classifier'] == cluster_no, 'swc']), alpha=1, linewidth=0.5,
                         method='2d', view='xy', group_neurons=True, ax=ax[x_ax][y_ax])
            ax[x_ax][y_ax].set_aspect('equal')
            ax[x_ax][y_ax].axis('off')
            ax[x_ax][y_ax].title.set_text(f'Cluster{cluster_no}')
            x_ax += 1

            if x_ax == no_of_ax:
                x_ax = 0
                y_ax += 1
        x_ax = 0
        y_ax = 0
        for cluster_no in range(no_of_ax):
            ax[x_ax][y_ax].axis('off')
            x_ax += 1

            if x_ax == no_of_ax:
                x_ax = 0
                y_ax += 1

        plt.title('all em neurons clustered')
        plt.show()
        all_cells_em = all_cells_em.sort_values('nblast_classifier')

    #initial nblast
    print('Start nblast')
    nb_all = nblast_two_groups(all_cells_em, all_cells_pa)

    #nblast on ipsi and contralateral
    print('Start nblast')
    nb_ipsi = nblast_two_groups(all_cells_em.loc[~all_cells_em['contralateral'], :], all_cells_pa.loc[all_cells_pa['morphology'] == 'ipsilateral', :])
    print('Start nblast')
    nb_contra = nblast_two_groups(all_cells_em.loc[all_cells_em['contralateral'],:],all_cells_pa.loc[all_cells_pa['morphology']=='contralateral',:])


    def plot_nblast_matrix(nb_df ,
        df1 ,
        df2 ,
        do_best_em_match4pa = True,
        do_best_pa_match4em = True,
        do_SymLogNorm = False,
        figsize = (20, 20),
        v_limit = 0.4,
        normalize = False,
        use_gregor_classifier = False,title='nblast matrix'):




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

        ax.plot([-1, -1], [0, nb_df.shape[0]], c='gray', lw=8, solid_capstyle='butt',alpha=0.5)
        if use_gregor_classifier:

            for dl in np.unique(df1.classifier):



                min = np.min(df1.loc[df1['classifier'] == dl, :].index)
                max = np.max(df1.loc[df1['classifier'] == dl, :].index)+1

                if min!=0:
                    ax.plot([-1.5, -0.5], [min, min], c='k', lw=2, solid_capstyle='butt',alpha=1)
                if max != nb_df.shape[0]:
                    ax.plot([-1.5, -0.5], [max, max], c='k', lw=2, solid_capstyle='butt',alpha=1)
                ax.text(-2,np.mean([min,max]),dl, horizontalalignment='right')
        else:

            for dl in np.unique(df1.nblast_classifier):



                min = np.min(df1.loc[df1['nblast_classifier'] == dl, :].index)
                max = np.max(df1.loc[df1['nblast_classifier'] == dl, :].index)+1

                if min!=0:
                    ax.plot([-1.5, -0.5], [min, min], c='k', lw=2, solid_capstyle='butt',alpha=1)
                if max != nb_df.shape[0]:
                    ax.plot([-1.5, -0.5], [max, max], c='k', lw=2, solid_capstyle='butt',alpha=1)
                ax.text(-2,np.mean([min,max]),dl, horizontalalignment='right')




        fig.colorbar(cbar)
        plt.suptitle(title)
        plt.show()


    plot_nblast_matrix(nb_contra, all_cells_em.loc[all_cells_em['contralateral'], :], all_cells_pa.loc[all_cells_pa['morphology'] == 'contralateral', :], title='nblast contralateral')
    plot_nblast_matrix(nb_ipsi,all_cells_em.loc[~all_cells_em['contralateral'], :],all_cells_pa.loc[all_cells_pa['morphology'] == 'ipsilateral', :],title='nblast ipsilateral')
    plot_nblast_matrix(nb_all, all_cells_em, all_cells_pa, title='nblast all')


    #overview matrices

    nb_all_sort_clustering = nb_all.loc[all_cells_em.sort_values('nblast_classifier')['cell_name'],:]

    mean_dt = nb_all_sort_clustering.loc[:, all_cells_pa.loc[all_cells_pa['function'] == 'dynamic threshold', 'cell_name']]
    mean_mc = nb_all_sort_clustering.loc[:, all_cells_pa.loc[all_cells_pa['function'] == 'motor command', 'cell_name']]
    mean_ci = nb_all_sort_clustering.loc[:, (all_cells_pa.loc[(all_cells_pa['function'] == 'integrator')&
                                               (all_cells_pa['morphology'] == 'contralateral'), 'cell_name'])]
    mean_ii = nb_all_sort_clustering.loc[:, (all_cells_pa.loc[(all_cells_pa['function'] == 'integrator') &
                                              (all_cells_pa['morphology'] == 'ipsilateral'), 'cell_name'])]

    median_dt = mean_dt.T.median()
    median_mc = mean_mc.T.median()
    median_ci = mean_ci.T.median()
    median_ii = mean_ii.T.median()
    
    mean_dt = np.mean(mean_dt,axis=1)
    mean_mc = np.mean(mean_mc,axis=1)
    mean_ci = np.mean(mean_ci,axis=1)
    mean_ii = np.mean(mean_ii,axis=1)




    mean_df = pd.concat([mean_dt,mean_mc,mean_ci,mean_ii],axis=1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
    lll = ax.pcolormesh(mean_df)
    fig.colorbar(lll)


    ax.axis('off')
    plt.show()
    
    median_df = pd.concat([median_dt,median_mc,median_ci,median_ii],axis=1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
    lll = ax.pcolormesh(median_df)
    fig.colorbar(lll)


    ax.axis('off')
    plt.show()


    import plotly
    brain_meshes = load_brs(path_to_data, 'raphe')
    cell = all_cells_pa.loc[38,'swc']

    fig = navis.plot3d([cell]+brain_meshes, backend='plotly',
                       width=1920, height=1080, hover_name=True, colors='purple')
    fig.update_layout(
        scene={
            'xaxis': {'autorange': 'reversed'},  # reverse !!!
            'yaxis': {'autorange': True},

            'zaxis': {'autorange': True},
            'aspectmode': "data",
            'aspectratio': {"x": 1, "y": 1, "z": 1}
        }
    )

    plotly.offline.plot(fig, filename="test.html", auto_open=True, auto_play=False)


    def find_crossing_neurite(nodes_df):

        all_segments_dict = {}
        for i,cell in nodes_df.loc[nodes_df['type']=='end',:].iterrows():
            all_segments_dict[cell['node_id']] = []
            exit_var = False
            work_cell = cell
            while exit_var != 'branch' or exit_var != 'root':
                all_segments_dict[cell['node_id']].append(work_cell['node_id'])
                work_cell = nodes_df.loc[nodes_df['node_id']==work_cell['parent_id'],:]
                exit_var = work_cell['type']

        return all_segments_dict


    find_crossing_neurite(cell.nodes)


