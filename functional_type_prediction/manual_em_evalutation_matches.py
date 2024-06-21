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


    #label neurotransmitter in name

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



    brs2load = ['Midbrain',"Forebrain","eye1","eye2","Hindbrain"]
    path2brs = path_to_data / "zbrain_regions" / "whole_brain"

    meshes = [navis.read_mesh(path2brs.joinpath(x+".obj"),units='um',output='volume') for x in brs2load]
    brain_mesh = navis.Volume.combine(meshes)


    path2save = path_to_data / 'make_figures_FK_output' / 'em_single_neurons_interactive_matches'
    matches_df = pd.read_excel(path_to_data / "em_zfish1" / 'manual-cell-assignment_em2paGFP.xlsx')
    matches_df = matches_df.iloc[:,:6]
    os.makedirs(path2save,exist_ok=True)


    for i,cell in  all_cells_em.iterrows():
        if not matches_df.loc[matches_df['EM cell']==cell['cell_name'],:].empty:
            matching_paGFP_list = matches_df.loc[matches_df['EM cell']==cell['cell_name'],'assigned paGFP cell(s)'].iloc[0]
        else:
            matching_paGFP_list = matches_df.loc[matches_df['EM cell'] == int(cell['cell_name']), 'assigned paGFP cell(s)'].iloc[0]
        if type(matching_paGFP_list) == str:
            matching_paGFP_list = matching_paGFP_list.rstrip().lstrip()
            matching_paGFP_list = matching_paGFP_list.replace(' ','')
            if matching_paGFP_list[-1] == ',':
                matching_paGFP_list = matching_paGFP_list[:-1]

            matching_paGFP_list = matching_paGFP_list.split(',')




            fig = navis.plot3d(brain_mesh,width=2500,height=1300,hover_name=True,backend='plotly',alpha=0.1)
            fig = navis.plot3d(cell['swc'],width=2500,height=1300,hover_name=True,backend='plotly',fig=fig,color='cyan')
            fig = navis.plot3d(navis.NeuronList(all_cells_pa.loc[all_cells_pa['cell_name'].isin(matching_paGFP_list),'swc']), width=2500, height=1300, hover_name=True, backend='plotly', fig=fig,color='red')

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


            plotly.offline.plot(fig, filename=f"{path2save}/{cell['cell_name']}_matches.html", auto_open=False, auto_play=False)
