import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
def find_connector_node_id(cell,all_syn):
    synapse_loc_array = np.array(all_syn.loc[:,['x','y','z']])
    node_loc_array = np.array(cell['swc'].nodes.loc[:,['node_id','x','y','z']])
    kd_tree = KDTree(node_loc_array[:,1:])
    distances, indices = kd_tree.query(synapse_loc_array)
    node_ids = node_loc_array[indices,0].astype(int)
    return node_ids


def load_synapse_clem(cell,path,mapped=True):

    if type(cell.cell_name) == pd.Series:
        cell_name = cell.cell_name.iloc[0]
    else:
        cell_name = cell.cell_name

    cell_name = f'clem_zfish1_{cell_name}'

    if mapped:
        path_post = path / 'clem_zfish1' / 'all_cells' / cell_name / 'mapped' / (cell_name + '_postsynapses_mapped.csv')
        path_pre = path / 'clem_zfish1' / 'all_cells' / cell_name / 'mapped' / (cell_name + '_presynapses_mapped.csv')
    else:
        path_post = path / 'clem_zfish1' / 'all_cells' / cell_name /  (cell_name + '_postsynapses.csv')
        path_pre = path / 'clem_zfish1' / 'all_cells' / cell_name /  (cell_name + '_presynapses.csv')

    list_of_synapses = []
    # Pull postsynapses
    if path_post.exists():
        post = pd.read_csv(path_post, comment='#', sep=' ', header=None, names=["connector_id", "x", "y", "z", "radius"])
        post.insert(1, 'type', 'post')
        list_of_synapses.append(post)
        # Pull presynapses

    if path_pre.exists():
        pre = pd.read_csv(path_pre, comment='#', sep=' ', header=None, names=["connector_id", "x", "y", "z", "radius"])
        pre.insert(1, 'type', 'pre')
        list_of_synapses.append(pre)
    try:
        all_syn = pd.concat(list_of_synapses, axis=0)
    except:
        pass
    try:
        all_syn = all_syn.reset_index(drop=True)
    except:
        pass
    try:
        all_syn.insert(1, 'node_id', find_connector_node_id(cell,all_syn))
        cell['swc'].connectors = all_syn


    except:
        pass

    try:
        cell['all_mesh'].connectors = all_syn
    except:
        pass



    return cell