import os

import matplotlib.pyplot as plt
import scipy
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells_predictor_pipeline import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.make_dendrogramms import *
import winsound

from datetime import datetime
import plotly
import matplotlib
import plotly.graph_objects as go



if __name__ == "__main__":



    name_time = datetime.now()
    # Set the base path for data by reading from a configuration file; ensures correct data location is used.
    path_to_data = get_base_path()  # Ensure this path is set in path_configuration.txt

    all_cells_em = load_cells_predictor_pipeline(path_to_data=path_to_data,modalities=["em"])
    all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"])
    for i,cell in all_cells_pa.iterrows():
        all_cells_pa.loc[i,'swc'].nodes.loc[:,"radius"] = 0.5
        all_cells_pa.loc[i, 'swc'].nodes.loc[0, "radius"] = 2
        all_cells_pa.loc[i,'swc'].soma = 1
        all_cells_pa.loc[i, 'swc'].name = all_cells_pa.loc[i,'swc'].name.split('_smoothed')[0]
    for i, cell in all_cells_em.iterrows():
        all_cells_em.loc[i, 'swc'].name = all_cells_em.loc[i, 'swc'].name.split('_')[2]



    brs2load = ['Midbrain',"Forebrain","eye1","eye2","Hindbrain"]
    path2brs = path_to_data / "zbrain_regions" / "whole_brain_copy"

    meshes = [navis.read_mesh(path2brs.joinpath(x+".obj"),units='um',output='volume') for x in brs2load]
    brain_mesh = navis.Volume.combine(meshes)


    path2save = path_to_data / 'make_figures_FK_output' / 'em_single_neurons_interactive'
    os.makedirs(path2save,exist_ok=True)

    for i,cell in  all_cells_em.iterrows():

        fig = navis.plot3d(brain_mesh,width=2500,height=1300,hover_name=True,backend='plotly',alpha=0.1)
        fig = navis.plot3d(cell['swc'],width=2500,height=1300,hover_name=True,backend='plotly',fig=fig,color='cyan')
        fig = navis.plot3d(navis.NeuronList(all_cells_pa['swc']), width=2500, height=1300, hover_name=True, backend='plotly', fig=fig,color='red')

        fig.update_layout(
            scene={
                'xaxis': {'autorange': 'reversed', 'zeroline': False, 'visible': False},  # reverse !!!
                'yaxis': {'autorange': True, 'zeroline': False, 'visible': False},

                'zaxis': {'autorange': True, 'zeroline': False, 'visible': False},
                'aspectmode': "data",
                'aspectratio': {"x": 1, "y": 1, "z": 1},
                'bgcolor': "black",
                "camera": {'projection': {'type':"perspective"}}

            }
        )


        plotly.offline.plot(fig, filename=f"{path2save}/{cell['cell_name']}.html", auto_open=False, auto_play=False)

    all_cells = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["em","pa","clem"],mirror=False)
    all_cells = all_cells.loc[~all_cells['swc'].isna(),:]
    for i,cell in all_cells.iterrows():
        all_cells.loc[i,'swc'].nodes.loc[:,"radius"] = 0.5
        all_cells.loc[i, 'swc'].nodes.loc[0, "radius"] = 2
        all_cells.loc[i,'swc'].soma = 1
        all_cells.loc[i, 'swc'].name = all_cells.loc[i,'swc'].name.split('_smoothed')[0]

    fig = navis.plot3d(brain_mesh, width=2500, height=1300, hover_name=True, backend='plotly', alpha=0.1)
    navis.plot3d(navis.NeuronList(all_cells.loc[all_cells['imaging_modality'] == "EM", 'swc']), width=2500, height=1300, hover_name=True, backend='plotly', fig=fig,alpha=0.1)
    navis.plot3d(navis.NeuronList(all_cells.loc[all_cells['imaging_modality'] == "photoactivation", 'swc']), width=2500, height=1300, hover_name=True, backend='plotly', fig=fig,alpha=0.1)
    navis.plot3d(navis.NeuronList(all_cells.loc[all_cells['imaging_modality'] == "clem", 'swc']), width=2500, height=1300, hover_name=True, backend='plotly', fig=fig,alpha=0.1)

    fig.update_layout(
        scene={
            'xaxis': {'autorange': 'reversed', 'zeroline': False, 'visible': False},  # reverse !!!
            'yaxis': {'autorange': True, 'zeroline': False, 'visible': False},

            'zaxis': {'autorange': True, 'zeroline': False, 'visible': False},
            'aspectmode': "data",
            'aspectratio': {"x": 1, "y": 1, "z": 1},
            'bgcolor': "black",
            "camera": {'projection': {'type': "perspective"}}

        }
    )
    plotly.offline.plot(fig, filename=f"{path2save}/all_cells.html", auto_open=True, auto_play=False)