import os
import matplotlib.pyplot as plt
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
matplotlib.use('TkAgg')

#path settings

path_to_data =  get_base_path() #path to clone of nextcloud, set your path in path_configuration.txt


#settings

modalities = ['clem','pa'] #modalities you want to query for your figure
keywords = ['integrator','contralateral'] #keywords of cells you want to visualize

#load pa  table
if 'pa' in modalities:
    pa_table = load_pa_table(path_to_data.joinpath("paGFP").joinpath("photoactivation_cells_table.csv"))
#load clem table
if 'clem' in modalities:
    clem_table = load_clem_table(path_to_data.joinpath('clem_zfish1').joinpath('all_cells'))

#TODO here the loading of gregor has to go


#concat tables
all_cells = pd.concat([eval(x+'_table') for x in modalities])
all_cells = all_cells.reset_index(drop=True)

#subset dataset for keywords
for keyword in keywords:
    subset_for_keyword = all_cells['cell_type_labels'].apply(lambda current_label: True if keyword.replace("_"," ") in current_label else False)
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


color_cell_type_dict = {"integrator":"red",
                        "dynamic threshold":"blue",
                        "motor command":"purple"}



#load needed meshes into a list and assign colors
#TODO this is stupid and convoluted, replace underscores cell trype labels for labels with spaces so datasets confirm with each other
color_cells = []
visualized_cells = []
for i,cell in all_cells.iterrows():

           for label in cell.cell_type_labels:
               if label.replace("_"," ") in color_cell_type_dict.keys():
                   temp_color = color_cell_type_dict[label.replace("_"," ")]
                   break
           for key in ["soma_mesh", "axon_mesh", "dendrite_mesh", "neurites_mesh"]:
                if not type(cell[key]) == float:
                   visualized_cells.append(cell[key])
                   if key != "dendrite_mesh":
                       color_cells.append(temp_color)
                   elif key == "dendrite_mesh":
                       color_cells.append("black")



#here we start the plotting
brain_meshes = load_brs(path_to_data,load_FK_regions=True)
selected_meshes = ["Retina", 'Midbrain', "Forebrain", "Habenula", "Hindbrain", "Spinal Cord"]
brain_meshes = [mesh for mesh in brain_meshes if mesh.name in selected_meshes]
color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)





fig = navis.plot3d(visualized_cells + brain_meshes, backend='plotly', color=color_cells + color_meshes, width=1920, height=1080)
fig.update_layout(
    scene={
        'xaxis': {'autorange': 'reversed', 'range': (0, 621 * 0.798)},  # reverse !!!
        'yaxis': {'range': (0, 1406 * 0.798)},

        'zaxis': {'range': (0, 138 * 2)},
    }
)

os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("html"),exist_ok=True)

#plotly.offline.plot(fig, filename=str(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("html").joinpath("test.html")), auto_open=True, auto_play=False)

# zprojection

fig, ax = navis.plot2d(visualized_cells,color=color_cells,alpha=1,linewidth=0.5,method='2d',view=('x', "-y"))

fig, ax = navis.plot2d(brain_meshes,
                       linewidth=0.5,
                       ax=ax,
                       alpha=0.2,
                       c=color_meshes,
                       method='2d',
                       view=('x', "-y"),
                       group_neurons=True,
                       # volume_outlines=False,
                       rasterize=False)

os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("pdf"),exist_ok=True)
os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("png"),exist_ok=True)
os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("svg"),exist_ok=True)
plt.savefig(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("pdf").joinpath(rf"z_projection_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=300)
plt.savefig(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("png").joinpath(rf"z_projection_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"), dpi=300)
plt.savefig(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("svg").joinpath(rf"z_projection_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=300)

