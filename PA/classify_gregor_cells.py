import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import navis
from pathlib import Path
import pandas as pd
from hindbrain_structure_function.visualization.FK_tools.load_pa_table import *
from hindbrain_structure_function.visualization.FK_tools.load_clem_table import *
from hindbrain_structure_function.visualization.FK_tools.load_mesh import *
from hindbrain_structure_function.visualization.FK_tools.load_brs import *
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
from datetime import datetime
import plotly
import matplotlib



name_time = datetime.now()
path_to_data = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data')
pa_table = load_pa_table(path_to_data.joinpath("paGFP").joinpath("photoactivation_cells_table.csv"))

all_pa_cells = pa_table

all_pa_cells['soma_mesh'] = np.nan
all_pa_cells['dendrite_mesh'] = np.nan
all_pa_cells['axon_mesh'] = np.nan
all_pa_cells['neurites_mesh'] = np.nan

all_pa_cells['soma_mesh'] = all_pa_cells['soma_mesh'].astype(object)
all_pa_cells['dendrite_mesh'] = all_pa_cells['dendrite_mesh'].astype(object)
all_pa_cells['axon_mesh'] = all_pa_cells['axon_mesh'].astype(object)
all_pa_cells['neurites_mesh'] = all_pa_cells['neurites_mesh'].astype(object)

for i, cell in all_pa_cells.iterrows():
    all_pa_cells.loc[i, :] = load_mesh(cell, path_to_data, use_smooth_pa=True)

#load gregors cells
#data seed cells
path_seed_cells = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\em_zfish1\data_seed_cells\output_data')
path_postsynaptic = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\em_zfish1\data_cell_89189_postsynaptic_partners\output_data')

em_cells = []
seed_or_post = []

for cell in os.listdir(path_seed_cells):
    path_to_cell = path_seed_cells.joinpath(cell).joinpath('mapped').joinpath(rf"em_fish1_{cell.split('_')[-1]}_mapped.obj")
    if path_to_cell.exists():
        em_cells.append(navis.read_mesh(path_to_cell, units="um"))
        seed_or_post.append('seed')
    else:
        print(cell, "not mapped")


for cell in os.listdir(path_postsynaptic):
    path_to_cell = path_postsynaptic.joinpath(cell).joinpath('mapped').joinpath(rf"em_fish1_{cell.split('_')[-1]}_mapped.obj")
    if path_to_cell.exists():
        em_cells.append(navis.read_mesh(path_to_cell, units="um"))
        seed_or_post.append('post')
    else:
        print(cell, "not mapped")


color_cell_type_dict = {"integrator": "red",
                        "dynamic threshold": "cyan",
                        "motor command": "purple",
                        "motor_command": "purple",
                        "dynamic_threshold": "cyan",}

visualized_cells = []
color_list = []

for i,cell in all_pa_cells.iterrows():
    visualized_cells.append(cell['neurites_mesh'])
    # visualized_cells.append(cell['soma_mesh'])
    temp = [x in color_cell_type_dict.keys() for x in cell['cell_type_labels'] ]
    color_list.append(color_cell_type_dict[(np.array(cell['cell_type_labels'])[temp])[0]])
    # color_list.append(color_cell_type_dict[(np.array(cell['cell_type_labels'])[temp])[0]])

for cell in em_cells:
    visualized_cells.append(cell)
    color_list.append('k')



fig = navis.plot3d(visualized_cells, backend='plotly',color = color_list,width=1920, height=1080)
fig.update_layout(
    scene={
        'xaxis': {'autorange': True},  # reverse !!!
        'yaxis': {'autorange': True},

        'zaxis': {'autorange': True},
        'aspectmode': "data",
        'aspectratio': {"x": 1, "y": 1, "z": 1}
    }
)



plotly.offline.plot(fig, filename="test.html", auto_open=True,
                    auto_play=False)

