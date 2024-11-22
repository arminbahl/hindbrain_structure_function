import navis
import numpy as np
import tifffile as tiff
import plotly
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
from collections import Counter
from typing import List, Deque
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm
from hindbrain_structure_function.functional_type_prediction.FK_tools.branching_angle_calculator import *
import plotly
from hindbrain_structure_function.functional_type_prediction.FK_tools.fragment_neurite import *
import sys

def repair_indices(nodes_df):
    it = 0
    for i,cell in nodes_df.iterrows():

        nodes_df.loc[nodes_df["parent_id"]==cell['node_id'],'parent_id'] = it
        nodes_df.loc[i,'node_id'] =it
        it += 1
    return nodes_df

def nodes2array(nodes_df):
    nodes_array = nodes_df['node_id'].to_numpy().astype(int)
    label_array = nodes_df['label'].to_numpy().astype(int)
    x_array = nodes_df['x'].to_numpy()
    y_array = nodes_df['y'].to_numpy()
    z_array = nodes_df['z'].to_numpy()
    radius_array = nodes_df['radius'].to_numpy()
    parent_array = nodes_df['parent_id'].to_numpy()

    type_array = nodes_df['type']

    a = np.stack([nodes_array,label_array,x_array, y_array, z_array,radius_array, parent_array,type_array], axis=1)
    return a


class dfs_Solution:
    # Topo sort only exists in DAGs i.e.
    # Direct Acyclic graph
    def dfs(self, adj, vis, node, n, stck):
        vis[node] = 1
        for it in adj[node]:
            if not vis[it]:
                self.dfs(adj, vis, it, n, stck)
        stck.append(node)

    # During the traversal u must
    # be visited before v
    def topo_sort(self, adj, n):
        vis = [0] * n

        # using stack ADT
        stck = []
        for i in range(n):
            if not vis[i]:
                self.dfs(adj, vis, i, n, stck)
        return stck


# Function to add an edge
def addEdge(adj: List[int], u: int, v: int) -> None:
    adj[u].append(v)


def repair_hierarchy(df):

    #create adjency
    n = df.shape[0]

    adj = [[] for _ in range(n)]
    for unique_parent in df.loc[:,'parent_id']:
        a_with_certain_parent = df.loc[df['parent_id']==unique_parent,:]
        for i,node in a_with_certain_parent.iterrows():
            if node['parent_id'] != -1:
                addEdge(adj, node['node_id'], node['parent_id'])


    s = dfs_Solution()
    try:
        sys.setrecursionlimit(3000)
        solution = s.topo_sort(adj, n)
    except:
        sys.setrecursionlimit(30000)
        solution = s.topo_sort(adj, n)


    new_id_assignment = {-1:-1}
    for i,new_assign in zip(range(df.shape[0]),solution):
        new_id_assignment[new_assign] = i

    df.loc[:,'node_id'] = df.loc[:,'node_id'].apply(lambda x: new_id_assignment[x])
    df.loc[:,'parent_id'] = df.loc[:,'parent_id'].apply(lambda x: new_id_assignment[x])
    df = df.sort_values('parent_id')

    return df

def check_neuron_by_viz(df_fixed, df_original,only_fragment=False):

    # fragment
    fragments = fragment_neuron_into_segments(df_fixed)
    all_fragments = []
    for key in fragments.keys():
        temp_df = df_fixed.loc[df_fixed['node_id'].isin(fragments[key]), :]
        temp_df.loc[~temp_df['parent_id'].isin(list(temp_df['node_id'])), 'parent_id'] = -1

        all_fragments.append(navis.TreeNeuron(temp_df, name=key))

    fixed_neuron = navis.TreeNeuron(df_fixed)
    fixed_neuron.nodes.loc[fixed_neuron.nodes['parent_id'] == -1, 'radius'] = 1
    fig = navis.plot3d(all_fragments, backend='plotly',
                       width=1920, height=1080, hover_name=True)
    if not only_fragment:
        fig = navis.plot3d(fixed_neuron, backend='plotly', fig=fig,
                           width=1920, height=1080, hover_name=True)

        fig = navis.plot3d(navis.TreeNeuron(df_original), backend='plotly', fig=fig,
                           width=1920, height=1080, hover_name=True, colors='green')


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

def repair_neuron(navis_element,path=None,viz_check=False):
    plot = False
    if navis_element.nodes.loc[(navis_element.nodes['radius']==2)&(navis_element.nodes['parent_id']==-1),:].empty:
        plot=True
        fig = navis.plot3d(navis_element, backend='plotly',
                           width=1920, height=1080, hover_name=True, alpha=1)




        temp_node_id = navis_element.nodes.loc[(navis_element.nodes['radius']==2),'node_id'].values[0]
        navis_element = navis_element.reroot(temp_node_id)
        print('REROOT',navis_element.name,'REROOT')
        fig.update_layout(
            scene={
                'xaxis': {'autorange': 'reversed'},  # reverse !!!
                'yaxis': {'autorange': True},

                'zaxis': {'autorange': True},
                'aspectmode': "data",
                'aspectratio': {"x": 1, "y": 1, "z": 1}
            }
        )


    df = repair_indices(navis_element.nodes)


    df = repair_hierarchy(df)

    df = df.iloc[:,:-1]
    #write neuron
    header = (f"# SWC format file based on specifications at http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html\n"
              f"# Generated by 'map_and_skeletonize_cell' of the ANTs registration helper library developed by the Bahl lab Konstanz.\n"
              f"# Labels: 0 = undefined; 1 = soma; 2 = axon; 3 = dendrite; 4 = Presynapse; 5 = Postsynapse\n")
    if path == None:
        with open("test.swc", 'w') as fp:
            fp.write(header)
            df.to_csv(fp, index=False, sep=' ', header=None)
    else:
        with open(path, 'w') as fp:
            fp.write(header)
            df.to_csv(fp, index=False, sep=' ', header=None)
    if plot and viz_check:
        ttt = navis.read_swc(path)
        ttt.soma = ttt.nodes.loc[(ttt.nodes['parent_id']==-1),'node_id'].values[0]
        ttt.name  = ttt.name+ " repaired"
        fig = navis.plot3d(ttt, backend='plotly', fig=fig,
                           width=1920, height=1080, hover_name=True, alpha=1)

        plotly.offline.plot(fig, filename="test.html", auto_open=True, auto_play=False)

if __name__ == '__main__':
    name_time = datetime.now()
    # set path
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')   # Ensure this path is set in path_configuration.txt
    path_to_data = Path(r'D:\hindbrain_structure_function\nextcloud')
    #load em data
    # all_cells_em = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["em"],mirror=False,load_repaired=False)
    # all_cells_em = all_cells_em.sort_values('classifier')
    #
    # #load pa cells
    # all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"],use_smooth=False)
    # all_cells_pa_smooth = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"],use_smooth=True)
    # all_cells_pa.loc[:,'swc_smooth'] = all_cells_pa_smooth['swc']
    # all_cells_pa = all_cells_pa.dropna(subset='swc',axis=0)
    #
    # #load clem cells
    # all_cells_clem = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["clem"],mirror=False,load_repaired=False)
    # all_cells_clem = all_cells_clem.dropna(subset='swc',axis=0)
    # all_cells_clem_predict = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["clem_predict"], mirror=False, load_repaired=False)
    # all_cells_clem_predict = all_cells_clem_predict.dropna(subset='swc', axis=0)
    #
    # all_cells = pd.concat([all_cells_clem_predict,all_cells_em,all_cells_clem,all_cells_pa])
    all_cells = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["clem"], mirror=False, load_repaired=False)
    all_cells = all_cells.dropna(subset='swc', axis=0)
    #repair all_swcs

    for i,cell in tqdm(all_cells.iterrows(),total=all_cells.shape[0]):

        if cell['imaging_modality'] == 'clem':


            temp_path = path_to_data /'clem_zfish1'/'all_cells_repaired'
            os.makedirs(temp_path,exist_ok=True)
            repair_neuron(cell['swc'],path=temp_path / f'clem_zfish1_{cell.cell_name}_repaired.swc')

        elif cell['imaging_modality'] == 'EM':
            temp_path = path_to_data  /'em_zfish1'/'all_cells_repaired'
            os.makedirs(temp_path,exist_ok=True)
            repair_neuron(cell['swc'], path=temp_path / f'em_zfish1_{cell.cell_name}_repaired.swc')

        elif cell['imaging_modality'] == 'photoactivation':
            temp_path = path_to_data  /'paGFP'/'all_cells_repaired'
            os.makedirs(temp_path,exist_ok=True)
            repair_neuron(cell['swc'], path=temp_path / f'{cell.cell_name}_repaired_smoothed.swc')

        print(f"em_zfish1_{cell.cell_name}_repaired.swc finished")

