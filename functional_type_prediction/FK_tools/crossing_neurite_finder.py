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
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells_predictor_pipeline import *
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
    # set path
    path_to_data = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data')
    #load em data
    all_cells_em = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["em"])
    all_cells_em = all_cells_em.sort_values('classifier')
    #load pa cells
    all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"])
    all_cells_pa = all_cells_pa.dropna(subset='swc',axis=0)
    #all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_pa['swc']]

    cell = all_cells_pa.loc[38,'swc']
    def find_end_neurites(nodes_df):
        all_segments_dict = {}
        for i, node in nodes_df.loc[nodes_df['type'] == 'end', :].iterrows():
            all_segments_dict[node['node_id']] = []
            exit_var = False
            work_cell = node
            all_segments_dict[node['node_id']].append(int(work_cell['node_id']))
            while exit_var != 'branch' and exit_var != 'root':

                try:
                    work_cell = nodes_df.loc[nodes_df['node_id'] == work_cell['parent_id'].iloc[0], :]
                except:
                    work_cell = nodes_df.loc[nodes_df['node_id'] == work_cell['parent_id'], :]
                exit_var = work_cell['type'].iloc[0]
                all_segments_dict[node['node_id']].append(int(work_cell['node_id']))

        return all_segments_dict

    def fragment_neuron_into_segments(nodes_df):
        all_segments_dict = {}
        for i, node in nodes_df.loc[(nodes_df['type'] == 'end')|(nodes_df['type'] == 'branch'), :].iterrows():
            all_segments_dict[node['node_id']] = []
            exit_var = False
            work_cell = node
            all_segments_dict[node['node_id']].append(int(work_cell['node_id']))
            while exit_var != 'branch' and exit_var != 'root':

                try:
                    work_cell = nodes_df.loc[nodes_df['node_id'] == work_cell['parent_id'].iloc[0], :]
                except:
                    work_cell = nodes_df.loc[nodes_df['node_id'] == work_cell['parent_id'], :]
                exit_var = work_cell['type'].iloc[0]
                all_segments_dict[node['node_id']].append(int(work_cell['node_id']))

        return all_segments_dict



    ttt = find_end_neurites(cell.nodes)
    lll = fragment_neuron_into_segments(cell.nodes)

    end_fragments = []
    all_fragments = []

    for key in ttt.keys():
        temp_df = cell.nodes.loc[cell.nodes['node_id'].isin(ttt[key]),:]
        temp_df.loc[~temp_df['parent_id'].isin(list(temp_df['node_id'])), 'parent_id'] = -1

        end_fragments.append(navis.TreeNeuron(temp_df))

    for key in lll.keys():
        temp_df = cell.nodes.loc[cell.nodes['node_id'].isin(lll[key]),:]
        temp_df.loc[~temp_df['parent_id'].isin(list(temp_df['node_id'])), 'parent_id'] = -1

        all_fragments.append(navis.TreeNeuron(temp_df))


    import plotly
    brain_meshes = load_brs(path_to_data, 'raphe')
    cell = all_cells_pa.loc[38,'swc']

    fig = navis.plot3d(all_fragments+brain_meshes, backend='plotly',
                       width=1920, height=1080, hover_name=True)
    fig = navis.plot3d(cell, backend='plotly',fig=fig,
                       width=1920, height=1080, hover_name=True)

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
