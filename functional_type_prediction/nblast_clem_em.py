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
from hindbrain_structure_function.functional_type_prediction.nblast_matrix_navis import *
if __name__ == "__main__":



    smat_fish = load_zebrafish_nblast_matrix(return_smat_obj=True, prune=False, modalities=['clem','pa'])
    def symmetric_log_transform(x, linthresh=1):
        return np.sign(x) * np.log1p(np.abs(x / linthresh))
    name_time = datetime.now()
    # set path
    path_to_data = get_base_path()  # Ensure this path is set in path_configuration.txt
    #load em data
    all_cells_clem = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["clem"])
    all_cells_clem = all_cells_clem.loc[all_cells_clem['swc'].dropna().index, :]

    #load pa cells
    all_cells_em = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["em"])
    all_cells_em = all_cells_em.dropna(subset='swc',axis=0)

    cell_type_categories = {'morphology': ['ipsilateral', 'contralateral'],
                            'neurotransmitter': ['inhibitory', 'excitatory'],
                            'function': ['integrator', 'dynamic_threshold', 'dynamic threshold', 'motor_command', 'motor command']}


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

    all_cells_clem = all_cells_clem.sort_values(['function','morphology','neurotransmitter'])
    prune = False
    if prune:
        all_cells_clem.loc[:,'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_clem['swc']]
        all_cells_em.loc[:, 'swc'] = [navis.prune_twigs(x, 20, recursive=True) for x in all_cells_em['swc']]

    nb_all = nblast_two_groups_custom_matrix(all_cells_clem, all_cells_em,custom_matrix=smat_fish)
    fast_mc_nblast = nb_all.loc[["cell_576460752631366630", "cell_576460752680445826"], :]
    top_matches = navis.nbl.extract_matches(fast_mc_nblast, 3)
    nb_all2 = nblast_two_groups(all_cells_clem, all_cells_em)
    fast_mc_nblast2 = nb_all2.loc[["cell_576460752631366630", "cell_576460752680445826",], :]
    top_matches2 = navis.nbl.extract_matches(fast_mc_nblast2, 3)

    sim_cells =[]
    for topmatch in [top_matches,top_matches2]:
        for match in range(3):
            for i,cell in topmatch.iterrows():
                if cell[f'match_{match+1}'] not in sim_cells:
                    sim_cells.append(cell[f'match_{match+1}'])



    clem_cells = list(all_cells_clem.loc[all_cells_clem['cell_name'].isin(["cell_576460752631366630", "cell_576460752680445826"]),'swc'])
    em_cells = list(all_cells_em.loc[all_cells_em['cell_name'].isin(sim_cells),'swc'])
    path_to_data = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data')
    brain_meshes = load_brs(path_to_data, 'whole_brain')


    import plotly
    fig = navis.plot3d(list(all_cells_em.loc[["seed_cells" in str(x) for x in  all_cells_em['metadata_path']],'swc']), backend='plotly',
                       width=1920, height=1080, hover_name=True,colors='red')
    fig = navis.plot3d(brain_meshes, backend='plotly',fig=fig,
                       width=1920, height=1080, hover_name=True)
    fig = navis.plot3d(em_cells, backend='plotly',fig=fig,
                       width=1920, height=1080, hover_name=True,colors='blue')


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


