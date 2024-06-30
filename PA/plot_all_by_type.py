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
import plotly
from copy import deepcopy

path_to_data = get_base_path()  # Ensure this path is set in path_configuration.txt
all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"])

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


brain_meshes = load_brs(path_to_data,'raphe')






#ii

ci = list(all_cells_pa.loc[(all_cells_pa['function']=='integrator')&(all_cells_pa['morphology']=='contralateral'),'swc'])
ci = [navis.prune_twigs(x,20,recursive=True) for x in ci]

dt = list(all_cells_pa.loc[(all_cells_pa['function']=='dynamic_threshold')&(all_cells_pa['morphology']=='contralateral'),'swc'])
dt = [navis.prune_twigs(x,20,recursive=True) for x in dt]

mc = list(all_cells_pa.loc[(all_cells_pa['function']=='motor_command')&(all_cells_pa['morphology']=='contralateral'),'swc'])
mc = [navis.prune_twigs(x,20,recursive=True) for x in mc]

fig = navis.plot3d(ci + brain_meshes,backend='plotly',
                   width=1920, height=1080,hover_name =True,colors='red')

fig = navis.plot3d(dt,backend='plotly',fig=fig,
                   width=1920, height=1080,hover_name =True,colors='blue')

fig=navis.plot3d(mc,backend='plotly',fig=fig,
                   width=1920, height=1080,hover_name =True,colors='purple')
fig.update_layout(
    scene={
        'xaxis': {'autorange': 'reversed'},  # reverse !!!
        'yaxis': {'autorange': True},

        'zaxis': {'autorange': True},
        'aspectmode': "data",
        'aspectratio': {"x": 1, "y": 1, "z": 1}
    }
)

plotly.offline.plot(fig, filename="test.html", auto_open=True,auto_play=False)


#prune_experiments
test_cell = all_cells_pa.loc[(all_cells_pa['cell_name']=='20230418.1'),'swc'].iloc[0]
test_cell = [navis.prune_twigs(test_cell,20,recursive=True)]
# list_cells = list(all_cells_pa.loc[(all_cells_pa['function']=='dynamic_threshold'),'swc'])


fig = navis.plot3d(test_cell + brain_meshes, backend='plotly',
                   width=1920, height=1080,hover_name =True,colors='red')
fig.update_layout(
    scene={
        'xaxis': {'autorange': 'reversed'},  # reverse !!!
        'yaxis': {'autorange': True},

        'zaxis': {'autorange': True},
        'aspectmode': "data",
        'aspectratio': {"x": 1, "y": 1, "z": 1}
    }
)

plotly.offline.plot(fig, filename="dt.html", auto_open=True,
                    auto_play=False)

