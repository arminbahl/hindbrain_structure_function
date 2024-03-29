import matplotlib.pyplot as plt
import numpy as np
import navis
from pathlib import Path
import pandas as pd
from hindbrain_structure_function.visualization.FK_tools.load_pa_table import *
from hindbrain_structure_function.visualization.FK_tools.load_clem_table import *
from hindbrain_structure_function.visualization.FK_tools.load_mesh import *
from hindbrain_structure_function.visualization.FK_tools.load_brs import *
import plotly
#path settings
path_to_data = Path(r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data")  #path to clone of nextcloud


#settings

modalities = ['clem','pa'] #modalities you want to query for your figure
keywords = ['integrator','contralateral'] #keywords of cells you want to visualize

#load pa  table
if 'pa' in modalities:
    pa_table = load_pa_table(r"C:\Users\ag-bahl\Desktop\photoactivation_cells_table.csv")
#load clem table
if 'clem' in modalities:
    clem_table = load_clem_table(path_to_data.joinpath('clem_zfish1').joinpath('all_cells'))

#TODO here the loading of gregor has to go


#concat tables
all_cells = pd.concat([eval(x+'_table') for x in modalities])
all_cells = all_cells.reset_index(drop=True)

#subset dataset for keywords
for keyword in keywords:
    subset_for_keyword = all_cells['cell_type_labels'].apply(lambda current_label: True if keyword in current_label else False)
    all_cells = all_cells[subset_for_keyword]



all_cells['soma_mesh'] = np.nan
all_cells['dendrite_mesh'] = np.nan
all_cells['axon_mesh'] = np.nan
all_cells['neurites_mesh'] = np.nan

all_cells['soma_mesh'] = all_cells['soma_mesh'].astype(object)
all_cells['dendrite_mesh'] = all_cells['dendrite_mesh'].astype(object)
all_cells['axon_mesh'] = all_cells['axon_mesh'].astype(object)
all_cells['neurites_mesh'] = all_cells['neurites_mesh'].astype(object)

#set imaging modality to clem if jon scored it
all_cells.loc[all_cells['tracer_names']=='Jonathan Boulanger-Weill','imaging_modality'] = 'clem'

#load the meshes for each cell that fits queries in selected modalities
for i,cell in all_cells.iterrows():
    all_cells.loc[i,:] = load_mesh(cell,path_to_data)

#put all cell meshes into a long list
cell_list = []
color_cells = []
for i,cell in all_cells.iterrows():
    try:
       cell_list.append(cell['soma_mesh'])

    except:
        pass

    try:
       cell_list.append(cell['neurites_mesh'])

    except:
        pass

    try:
       cell_list.append(cell['axon_mesh'])

    except:
        pass

    try:
       cell_list.append(cell['dendrite_mesh'])

    except:
        pass

    if cell['imaging_modality'] == 'photoactivation':
        color_cells.append('red')
        color_cells.append('red')

    if cell['imaging_modality'] == 'clem':
        color_cells.append('pink')
        color_cells.append('pink')
        color_cells.append('pink')



#here we start the plotting
brain_meshes = load_brs()
selected_meshes = ["Retina", 'Midbrain', "Forebrain", "Habenula", "Hindbrain", "Spinal Cord"]
brain_meshes = [mesh for mesh in brain_meshes if mesh.name in selected_meshes]
color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)





fig = navis.plot3d(cell_list + brain_meshes, backend='plotly', color=color_cells + color_meshes, width=2500, height=1300)
fig.update_layout(
    scene={
        'xaxis': {'autorange': 'reversed', 'range': (0, 621 * 0.798)},  # reverse !!!
        'yaxis': {'range': (0, 1406 * 0.798)},

        'zaxis': {'range': (0, 138 * 2)},
    }
)
plotly.offline.plot(fig, filename=rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\test.html", auto_open=True, auto_play=False)




