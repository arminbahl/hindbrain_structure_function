import os

import matplotlib.pyplot as plt
import scipy
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
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
    cells_clem1 = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["clem"])
    cells_clem2 = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['prediction_project'],
                                                 load_repaired=True)
    cells_clem3 = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem_predict'], load_repaired=True)
    all_cells_clem = pd.concat([cells_clem1, cells_clem2,cells_clem3])
    all_cells_clem = all_cells_clem.drop_duplicates(subset='cell_name')

    all_cells_clem['comment_mentioned'] = (all_cells_clem.comment.apply(lambda x: [x.split(' ')[i] for i in range(len(x.split(' '))) if x.split(' ')[i].isnumeric()]))
    cell_no_in_comments_dict = {}
    for i,cell in all_cells_clem.iterrows():
        if cell['comment_mentioned']:
            for mentioned_cell in cell['comment_mentioned']:
                if mentioned_cell  in np.unique(list(all_cells_clem.functional_id)):
                    if mentioned_cell not in cell_no_in_comments_dict:
                        cell_no_in_comments_dict[mentioned_cell] = 1
                    else:
                        cell_no_in_comments_dict[mentioned_cell] += 1


    #functional imaged connectome cells
    all_cells_clem = all_cells_clem.loc[all_cells_clem['functional_id'].isin([int(x) for x in cell_no_in_comments_dict.keys()]  ), :]

    #fast integrators
    # all_cells_clem = all_cells_clem.loc[all_cells_clem['cell_name'].isin(['cell_576460752680445826', "cell_576460752631366630"]), :]

    #only seed cells in EM
    all_cells_em = all_cells_em.loc[all_cells_em['seed_cells'],:]

    for i,cell in all_cells_clem.iterrows():
        all_cells_clem.loc[i,'swc'].nodes.loc[:,"radius"] = 0.5
        all_cells_clem.loc[i, 'swc'].nodes.loc[0, "radius"] = 2
        all_cells_clem.loc[i,'swc'].soma = 1
    for i, cell in all_cells_em.iterrows():
        all_cells_em.loc[i, 'swc'].name = all_cells_em.loc[i, 'swc'].name.split('_')[2]



    brs2load = ['Midbrain',"Forebrain","eye1","eye2","Hindbrain",'raphe']
    path2brs = path_to_data / "zbrain_regions" / "whole_brain"

    meshes = [navis.read_mesh(path2brs.joinpath(x+".obj"),units='um',output='volume') for x in brs2load]
    brain_mesh = navis.Volume.combine(meshes)


    path2save = path_to_data / 'make_figures_FK_output' / 'clem_matching2em'
    os.makedirs(path2save,exist_ok=True)

    for i,cell in  all_cells_clem.iterrows():

        fig = navis.plot3d(brain_mesh,width=2500,height=1300,hover_name=True,backend='plotly',alpha=0.1)
        fig = navis.plot3d(cell['swc'],width=2500,height=1300,hover_name=True,backend='plotly',fig=fig,color='cyan')
        fig = navis.plot3d(navis.NeuronList(all_cells_em['swc']), width=2500, height=1300, hover_name=True, backend='plotly', fig=fig,color='red')

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


        plotly.offline.plot(fig, filename=f"{path2save}/{cell['cell_name']}_{cell['functional_id']}.html", auto_open=False, auto_play=False)

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