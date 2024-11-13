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
import pickle

def calculate_zebrafish_nblast_matrix(df,with_kunst=True,return_smat_obj=False,prune=True,modalities=["pa"],path_to_data = None):
    if path_to_data is  None:
        path_to_data = get_base_path()
    kunst_neurons = []
    if with_kunst:
        with h5py.File(path_to_data / 'zbrain_regions' / 'all_mpin_single_cells.hdf5') as f:
            for key1 in f.keys():
                print(key1)
                for key2 in tqdm(f[key1].keys(),total=len(f[key1].keys())):
                    temp_df = pd.DataFrame(f[key1][key2])
                    temp_df.columns = ['node_id', 'label', 'x', 'y', 'z', 'radius', 'parent_id']
                    temp_df.loc[:,"x"] = temp_df.loc[:,"x"]*0.798
                    temp_df.loc[:, "y"] = temp_df.loc[:, "y"] *0.798
                    temp_df.loc[:, "z"] = temp_df.loc[:, "z"] *2
                    kunst_neurons.append(navis.TreeNeuron(temp_df,units='1 micrometer'))
    if with_kunst:
        all_neurons = np.concatenate([list(df.loc[:, 'swc'].values), kunst_neurons])
    else:
        all_neurons = np.array(list(df.loc[:, 'swc'].values))
    all_neurons = [navis.resample_skeleton(n, '1 micrometer') for n in all_neurons]
    all_neurons_name = [x.name for x in all_neurons]
    function_label = []
    for cell_name in all_neurons_name:
        if cell_name is None:
            function_label.append('na')
        else:
            if 'clem' in cell_name:
                temp = cell_name.split('_repaired')[0].split('zfish1_')[1]
            elif 'em' in cell_name:
                temp = cell_name.split('_repaired')[0].split('zfish1_')[1]
            else:
                temp = cell_name.split('_repaired')[0]

            temp_function = df.loc[df['cell_name']==temp,'function'].iloc[0]
            temp_morphology = df.loc[df['cell_name'] == temp, 'morphology'].iloc[0]
            if temp_function  in ['neg_control', 'to_predict']:
                temp_function = 'na'
            if temp_function == 'integrator':
                temp_function = temp_morphology + '_' + temp_function

            function_label.append(temp_function)




    matches = []
    for i1 in np.arange(len(function_label))[np.array(function_label)!='na']:
        for i2 in np.arange(len(function_label))[np.array(function_label)!='na']:
            if function_label[i1] == function_label[i2] and i1 != i2:
                matches.append([i1,i2])

    dotprops = [navis.make_dotprops(n, k=5, resample=False) for n in all_neurons]
    builder = LookupDistDotBuilder(dotprops, matches, use_alpha=True, seed=2021).with_bin_counts([10, 10])
    smat = builder.build()
    as_table = smat.to_dataframe()



    if return_smat_obj:
        return smat

    else:
        return as_table



if __name__ == "__main__":
    def symmetric_log_transform(x, linthresh=1):
        return np.sign(x) * np.log1p(np.abs(x / linthresh))

    # set path
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')   # Ensure this path is set in path_configuration.txt
    #load em data
    all_cells_clem = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["clem"])
    all_cells_clem = all_cells_clem
    all_cells_clem.loc[:, 'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_clem['swc']]
    all_cells_clem.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_clem['swc']]
    #load pa cells
    all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"],use_smooth=False)
    all_cells_pa_smooth = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"],use_smooth=True)
    all_cells_pa.loc[:,'swc'] = all_cells_pa_smooth['swc']
    all_cells_pa = all_cells_pa.dropna(subset='swc',axis=0)
    all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_pa['swc']]
    all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_pa['swc']]


    all_cells_pa = pd.concat([all_cells_pa, all_cells_clem])
    all_cells_pa = all_cells_clem

    kunst_neurons = []

    with h5py.File(r"Y:\Zebrafish atlases\z_brain_atlas\single_cells\all_mpin_single_cells.hdf5") as f:
        for key1 in f.keys():
            print(key1)
            for key2 in tqdm(f[key1].keys(),total=len(f[key1].keys())):
                temp_df = pd.DataFrame(f[key1][key2])
                temp_df.columns = ['node_id', 'label', 'x', 'y', 'z', 'radius', 'parent_id']
                temp_df.loc[:,"x"] = temp_df.loc[:,"x"]*0.798
                temp_df.loc[:, "y"] = temp_df.loc[:, "y"] *0.798
                temp_df.loc[:, "z"] = temp_df.loc[:, "z"] *2
                kunst_neurons.append(navis.TreeNeuron(temp_df))



    #make matrix

    original = list(all_cells_pa.loc[(all_cells_pa['function'] == 'integrator')&(all_cells_pa['morphology'] == 'ipsilateral'), 'swc'])
    all = list(all_cells_pa.loc[:, 'swc'])
    all = kunst_neurons

    dotprops = [navis.make_dotprops(n, k=5, resample=False) for n in original + all]
    matching_pairs = [[idx, idx + len(original)] for idx in range(len(original))]


    builder = LookupDistDotBuilder(dotprops, matching_pairs, use_alpha=True, seed=2021).with_bin_counts([15, 15])
    smat = builder.build()
    as_table = smat.to_dataframe()
    as_table

    plt.imshow(as_table)
    plt.title('Integrator ipsi')
    plt.colorbar()
    plt.show()


    import plotly
    path_to_data = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data')
    brain_meshes = load_brs(path_to_data, 'whole_brain')


    # fig = navis.plot3d(kunst_neurons+brain_meshes, backend='plotly',
    #                    width=1920, height=1080, hover_name=True)
    #
    #
    # fig.update_layout(
    #     scene={
    #         'xaxis': {'autorange': 'reversed'},  # reverse !!!
    #         'yaxis': {'autorange': True},
    #
    #         'zaxis': {'autorange': True},
    #         'aspectmode': "data",
    #         'aspectratio': {"x": 1, "y": 1, "z": 1}
    #     }
    # )
    #
    # plotly.offline.plot(fig, filename="test.html", auto_open=True, auto_play=False)


