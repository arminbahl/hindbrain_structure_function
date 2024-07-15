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
    # set path
    path_to_data = get_base_path()  # Ensure this path is set in path_configuration.txt
    #load em data
    all_cells_clem = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["clem"])
    all_cells_clem = all_cells_clem.loc[all_cells_clem['swc'].dropna().index, :]

    #load pa cells
    all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"])
    all_cells_pa = all_cells_pa.dropna(subset='swc',axis=0)

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

    for i, cell in all_cells_clem.iterrows():
        if type(cell.cell_type_labels) == list:
            for label in cell.cell_type_labels:
                if label in cell_type_categories['morphology']:
                    all_cells_clem.loc[i, 'morphology'] = label
                elif label in cell_type_categories['function']:
                    all_cells_clem.loc[i, 'function'] = label
                elif label in cell_type_categories['neurotransmitter']:
                    all_cells_clem.loc[i, 'neurotransmitter'] = label
        all_cells_clem.loc[i, 'swc'].name = all_cells_clem.loc[i, 'swc'].name + " NT: " + all_cells_clem.loc[i, 'neurotransmitter']
    all_cells_clem = all_cells_clem.loc[all_cells_clem['function']!='nan', :]
    all_cells_pa = all_cells_pa.sort_values(['function','morphology','neurotransmitter'])
    all_cells_clem = all_cells_clem.sort_values(['function','morphology','neurotransmitter'])
    prune = False
    if prune:
        all_cells_clem.loc[:,'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_clem['swc']]
        all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_pa['swc']]

    nb_all = nblast_two_groups(all_cells_clem, all_cells_pa)
    aaa = plt.pcolormesh(nb_all)
    plt.colorbar(aaa)
    plt.show()

