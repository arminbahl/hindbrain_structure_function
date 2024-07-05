import navis
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

def symmetric_log_transform(x, linthresh=1):
    return np.sign(x) * np.log1p(np.abs(x / linthresh))
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
def find_crossing_neurite(fragmented_dict,nodes_df):
    width_brain = 495.56
    key_crossing = None
    for key in fragmented_dict.keys():
        temp = nodes_df.loc[nodes_df['node_id'].isin(fragmented_dict[key]), 'x']
        #check if any branch has a node close to the midline
        if (temp>(width_brain/2)+2).any() and (temp<(width_brain/2)-2).any():
            key_crossing=key
            #exact crossing
            idx_crossing = abs(nodes_df.loc[nodes_df['node_id'].isin(fragmented_dict[key]),'x']-(width_brain/2)).argmin()
            node_id_crossing = nodes_df.loc[nodes_df['node_id'].isin(fragmented_dict[key]),:].iloc[idx_crossing].node_id
            coords_crossing = np.array(nodes_df.loc[nodes_df['node_id']==node_id_crossing,['x','y','z']])[0]

            #median crossing
            # x_cross = nodes_df.loc[(nodes_df['node_id'].isin(fragmented_dict[key]))&
            #                        ((nodes_df['x']>(width_brain/2)-2)|
            #                         (nodes_df['x']<(width_brain/2)+2)), 'x'].median()
            # y_cross = nodes_df.loc[(nodes_df['node_id'].isin(fragmented_dict[key])) &
            #                        ((nodes_df['x'] >  (width_brain / 2) - 2) |
            #                         (nodes_df['x'] <  (width_brain / 2) + 2)), 'y'].median()
            #
            # z_cross = nodes_df.loc[(nodes_df['node_id'].isin(fragmented_dict[key])) &
            #                        ((nodes_df['x'] >  (width_brain / 2) - 2) |
            #                         (nodes_df['x']  < (width_brain / 2) + 5)), 'z'].median()
            # coords_crossing = np.array([x_cross, y_cross, z_cross])

            print('Crossing neurite found!')
    if key_crossing == None:
        coords_crossing = np.nan
        key_crossing = np.nan
    return key_crossing,coords_crossing
def find_fragment_main_branching(fragmented_dict, current_fragment_key, target_fragment_key, visited=None,reverse=False):

    if reverse:
        for key in fragmented_dict.keys():
            fragmented_dict[key] = fragmented_dict[key][::-1]

    if visited is None:
        visited = set()

    if current_fragment_key in visited:
        return None

    visited.add(current_fragment_key)

    next_steps = []
    for key in fragmented_dict.keys():
        if key != current_fragment_key:
            if fragmented_dict[current_fragment_key][0] in fragmented_dict[key]:
                if key == target_fragment_key:
                    return [current_fragment_key, key]
                else:
                    next_steps.append(key)

    if not next_steps:
        return None
    else:
        for key in next_steps:
            result = find_fragment_main_branching(fragmented_dict, key, target_fragment_key, visited)
            if result is not None:
                return [current_fragment_key] + result

    return None
def alternative_find_fragment_main_branching(fragmented_dict, current_fragment_key, target_fragment_key, visited=None):
    if visited is None:
        visited = set()

    if current_fragment_key in visited:
        return None

    visited.add(current_fragment_key)

    next_steps = []
    for key in fragmented_dict.keys():
        if key != current_fragment_key:
            if fragmented_dict[current_fragment_key][0] in fragmented_dict[key]:
                if key == target_fragment_key:
                    return [current_fragment_key, key]
                else:
                    next_steps.append(key)

    if not next_steps:
        return None
    else:
        for key in next_steps:
            result = find_fragment_main_branching(fragmented_dict, key, target_fragment_key, visited)
            if result is not None:
                return [current_fragment_key] + result

    return None
def find_primary_path(fragmented_dict,path):
    primary = path[1]
    for i,key in enumerate(path[1:]):
        if fragmented_dict[path[0]][-1] in fragmented_dict[key]:
            primary = key


    return primary
def calculate_vector(nodes_df):
    a1 = np.array(nodes_df.iloc[0].loc[['x','y','z']])
    a2 = np.array(nodes_df.iloc[-1].loc[['x','y','z']])
    vector = a1 - a2
    return vector
def angle_between_vectors(branch1, branch2,against_z = False):
    v1 = calculate_vector(branch1)
    if against_z:
        v1 = np.array([0,0,1])
    v2 = calculate_vector(branch2)
    # Calculate the dot product of the vectors
    dot_product = np.dot(v1, v2)

    # Calculate the magnitudes of the vectors
    magnitude_a = np.linalg.norm(v1)
    magnitude_b = np.linalg.norm(v2)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_a * magnitude_b)

    # Calculate the angle in radians
    angle_radians = np.arccos(cos_theta)

    # Convert the angle to degrees (optional)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees
def repair_neuron(nodes_df):
    it = 0
    for i,cell in nodes_df.iterrows():
        it +=1
        nodes_df.loc[nodes_df["parent_id"]==cell['node_id'],'parent_id'] = it
        nodes_df.loc[i,'node_id'] =it
    return nodes_df
def fragmented_to_plot(df,fragments):
    all_fragments = []
    for key in fragments.keys():

        temp_df = df.loc[df['node_id'].isin(fragments[key]),:]
        temp_df.loc[~temp_df['parent_id'].isin(list(temp_df['node_id'])), 'parent_id'] = -1


        all_fragments.append(navis.TreeNeuron(temp_df,name=key))
        if -1 in list(all_fragments[-1].nodes['parent_id']):
            temp_node_id = all_fragments[-1].nodes.loc[all_fragments[-1].nodes['parent_id']==-1,'node_id'].iloc[0]
            all_fragments[-1].soma = temp_node_id
        if len(all_fragments) == 1:
            all_fragments[0].nodes.loc[all_fragments[0].nodes['parent_id']==-1,'radius'] = 2

    return all_fragments
def direct_angle_and_crossing_extraction(nodes_df,angle2zaxis=False):
    width_brain = 495.56
    fragments = fragment_neuron_into_segments(nodes_df)
    fragments_list = fragmented_to_plot(nodes_df, fragments)
    #nodes_df = repair_neuron(nodes_df)
    # if not (nodes_df.loc[:, "x"] > (width_brain / 2)+10).any():
    #     return np.nan, np.nan,fragments_list

    #extract all fragments

    #write into dict


    #extract crossing neurite
    crossing_key, crossing_coords = find_crossing_neurite(fragments, nodes_df)
    if crossing_key == None or np.isnan(crossing_key):

        return np.nan, np.nan,fragments_list
    else:
        possible_path = find_fragment_main_branching(fragments,list(fragments.keys())[0], crossing_key)
        if possible_path == None:
            possible_path  = find_fragment_main_branching(fragments,list(fragments.keys())[0], crossing_key,reverse=True)

        if possible_path == None:
            return np.nan, crossing_coords,fragments_list


        father_of_crossing_key = find_primary_path(fragments,possible_path)

        main_branch = nodes_df.loc[nodes_df['node_id'].isin(fragments[list(fragments.keys())[0]]), :]
        # main_branch.loc[~main_branch['parent_id'].isin(list(main_branch['node_id'])), 'parent_id'] = -1
        # main_branch = main_branch.iloc[[0, -1], :]
        # main_branch.loc[:, 'parent_id'].iloc[-1] = main_branch.iloc[0].loc['node_id']

        first_branch = nodes_df.loc[nodes_df['node_id'].isin(fragments[father_of_crossing_key]), :]
        # first_branch.loc[~first_branch['parent_id'].isin(list(first_branch['node_id'])), 'parent_id'] = -1
        # first_branch = first_branch.iloc[[0, -1], :]
        # first_branch.loc[:, 'parent_id'].iloc[-1] = first_branch.iloc[0].loc['node_id']
        #
        # main_branch_tree = navis.TreeNeuron(main_branch)
        # main_branch_tree.name = 'MAIN BRANCH'

        # first_branch_tree = navis.TreeNeuron(first_branch)
        # first_branch_tree.name = 'FIRST BRANCH'

        angle = angle_between_vectors(main_branch, first_branch,angle2zaxis)
        angle = round(angle, 2)

        return angle, crossing_coords,fragments_list



if __name__ == "__main__":
    #load cells
    name_time = datetime.now()
    # set path
    path_to_data = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data')
    #load em data
    all_cells_em = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["em"],load_repaired=True)
    all_cells_em = all_cells_em.sort_values('classifier')
    #load pa cells
    # all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"],use_smooth=False)
    # all_cells_pa_smooth = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"], use_smooth=True)
    # all_cells_pa.loc[:,"swc_smooth"] = all_cells_pa_smooth['swc']



    all_cells_em.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_em['swc']]

    nodes_df = all_cells_em.loc[38,'swc'].nodes


    width_brain = 495.56
    fragments = fragment_neuron_into_segments(nodes_df)
    crossing_key, crossing_coords = find_crossing_neurite(fragments, nodes_df)


    possible_path = find_fragment_main_branching(fragments, list(fragments.keys())[0], crossing_key)

    father_of_crossing_key = find_primary_path(fragments, possible_path)

    main_branch = nodes_df.loc[nodes_df['node_id'].isin(fragments[list(fragments.keys())[0]]), :]
    main_branch.loc[~main_branch['parent_id'].isin(list(main_branch['node_id'])), 'parent_id'] = -1

    main_branch_vector = main_branch.iloc[-20:,:]
    main_branch_vector.iloc[0,-1] = -1
    main_branch_vector = main_branch_vector.iloc[[0, -1], :]
    main_branch_vector.iloc[-1, -1] = main_branch_vector.iloc[0, 0]

    first_branch = nodes_df.loc[nodes_df['node_id'].isin(fragments[father_of_crossing_key]), :]
    first_branch.loc[~first_branch['parent_id'].isin(list(first_branch['node_id'])), 'parent_id'] = -1
    first_branch_vector = first_branch.iloc[:20,:]
    first_branch_vector = first_branch_vector.iloc[[0,-1],:]
    first_branch_vector.iloc[-1,-1] = first_branch_vector.iloc[0,0]




    main_branch_tree = navis.TreeNeuron(main_branch)
    main_branch_tree.name = 'MAIN BRANCH'

    first_branch_tree = navis.TreeNeuron(first_branch)
    first_branch_tree.name = 'FIRST '

    main_branch_tree_vector = navis.TreeNeuron(main_branch_vector)
    main_branch_tree_vector.name = 'MAIN BRANCH VECTOR'

    first_branch_vector_tree = navis.TreeNeuron(first_branch_vector)
    first_branch_vector_tree.name = 'FIRST VECTOR'

    angle = angle_between_vectors(main_branch, first_branch, False)
    angle = round(angle, 2)


    all_fragments = fragmented_to_plot(nodes_df,fragments)

    crossing_neurite_df = nodes_df.loc[nodes_df['node_id'].isin(fragments[crossing_key]),:]

    crossing_neurite_neuron = navis.TreeNeuron(crossing_neurite_df)
    crossing_neurite_neuron.nodes.iloc[0, -1] = -1

    crossing_coords_df = pd.DataFrame(crossing_coords).T
    crossing_coords_df.loc[0, 'node_id'] = 1
    crossing_coords_df.loc[0, 'parent_id'] = -1
    crossing_coords_df.columns = ['x', 'y', 'z', 'node_id', 'parent_id']






    import plotly

    path_to_data = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data')
    brain_meshes = load_brs(path_to_data, 'raphe')


    fig = navis.plot3d(all_fragments+brain_meshes, backend='plotly',
                       width=1920, height=1080, hover_name=True)
    fig = navis.plot3d(navis.TreeNeuron(nodes_df), backend='plotly',fig=fig,
                       width=1920, height=1080, hover_name=True)

    fig = navis.plot3d(crossing_neurite_neuron, backend='plotly', fig=fig,
                       width=1920, height=1080, hover_name=True)
    fig = navis.plot3d(main_branch_tree_vector, backend='plotly', fig=fig,
                       width=1920, height=1080, hover_name=True,colors='red')
    fig = navis.plot3d(first_branch_vector_tree, backend='plotly', fig=fig,
                       width=1920, height=1080, hover_name=True,colors='red')


    fig = navis.plot3d(crossing_coords_df, backend='plotly', fig=fig,
                       width=1920, height=1080, hover_name=True)

    fig = navis.plot3d(main_branch_tree, backend='plotly', fig=fig,
                       width=1920, height=1080, hover_name=True)

    fig = navis.plot3d(first_branch_tree, backend='plotly', fig=fig,
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



