import os

import matplotlib.pyplot as plt
import scipy
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells_predictor_pipeline import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.make_dendrogramms import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.find_branches import *
import winsound
from copy import deepcopy
from datetime import datetime
import plotly
import matplotlib
# matplotlib.use('TkAgg')
from tqdm import tqdm

if __name__ == "__main__":



    name_time = datetime.now()
    # Set the base path for data by reading from a configuration file; ensures correct data location is used.
    path_to_data = get_base_path()  # Ensure this path is set in path_configuration.txt
    all_cells = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=['pa', "em"])
    all_cells_shifted_neurons = deepcopy(all_cells)
    cell_type_categories = {'morphology':['ipsilateral','contralateral'],
                            'neurotransmitter':['inhibitory','excitatory'],
                            'function':['integrator','dynamic_threshold','dynamic threshold','motor_command','motor command']}

    for i,cell in all_cells.iterrows():
        if type(cell.cell_type_labels) == list:
            for label in cell.cell_type_labels:
                if label in cell_type_categories['morphology']:
                    all_cells.loc[i,'morphology'] = label
                elif label in cell_type_categories['function']:
                    all_cells.loc[i,'function'] = label
                elif label in cell_type_categories['neurotransmitter']:
                    all_cells.loc[i,'neurotransmitter'] = label



     #
    nb_df = nblast_two_groups(all_cells.loc[all_cells['imaging_modality']=='EM',:],all_cells.loc[all_cells['imaging_modality']!='EM',:])

    for i,cell in nb_df.iterrows():
        all_cells.loc[all_cells['cell_name']==i,"best_match"] = cell.index[cell.argmax()]
        all_cells.loc[all_cells['cell_name'] == i, "best_match_value"] = cell.max()
        all_cells.loc[all_cells['cell_name'] == i, "best_match_function"] = all_cells.loc[all_cells['cell_name'] == cell.index[cell.argmax()], "function"].iloc[0]
        all_cells.loc[all_cells['cell_name'] == i, "best_match_morphology"] = all_cells.loc[all_cells['cell_name'] == cell.index[cell.argmax()], "morphology"].iloc[0]
        all_cells.loc[all_cells['cell_name'] == i, "best_match_neurotransmitter"] = all_cells.loc[all_cells['cell_name'] == cell.index[cell.argmax()], "neurotransmitter"].iloc[0]

    grouped = all_cells.loc[all_cells['imaging_modality']=="EM",['best_match_function','best_match_morphology','cell_name','best_match_value','best_match']].groupby(['best_match_function','best_match_morphology','cell_name']).max()
    grouped = grouped.sort_values(['best_match_function','best_match_morphology'],ascending=False)
    
    
    
    all_cells_shifted_neurons = deepcopy(all_cells)
    for i, cell in all_cells_shifted_neurons.iterrows():
        x_shift = cell['swc'].nodes.loc[0,'x']
        y_shift = cell['swc'].nodes.loc[0, 'y']
        z_shift = cell['swc'].nodes.loc[0, 'z']
        all_cells_shifted_neurons.loc[i,'swc'].nodes.loc[:, 'x'] = cell['swc'].nodes.loc[:, 'x'] - x_shift
        all_cells_shifted_neurons.loc[i, 'swc'].nodes.loc[:, 'y'] = cell['swc'].nodes.loc[:, 'y'] - y_shift
        all_cells_shifted_neurons.loc[i, 'swc'].nodes.loc[:, 'z'] = cell['swc'].nodes.loc[:, 'z'] - z_shift
        
    nb2_df = nblast_two_groups(all_cells_shifted_neurons.loc[all_cells_shifted_neurons['imaging_modality']=='EM',:],all_cells_shifted_neurons.loc[all_cells_shifted_neurons['imaging_modality']!='EM',:])

    for i,cell in nb2_df.iterrows():
        all_cells.loc[all_cells['cell_name']==i,"shifted_best_match"] = cell.index[cell.argmax()]
        all_cells.loc[all_cells['cell_name'] == i, "shifted_best_match_value"] = cell.max()
        all_cells.loc[all_cells['cell_name'] == i, "shifted_best_match_function"] = all_cells.loc[all_cells['cell_name'] == cell.index[cell.argmax()], "function"].iloc[0]
        all_cells.loc[all_cells['cell_name'] == i, "shifted_best_match_morphology"] = all_cells.loc[all_cells['cell_name'] == cell.index[cell.argmax()], "morphology"].iloc[0]
        all_cells.loc[all_cells['cell_name'] == i, "shifted_best_match_neurotransmitter"] = all_cells.loc[all_cells['cell_name'] == cell.index[cell.argmax()], "neurotransmitter"].iloc[0]
        
    grouped_shifted = all_cells.loc[all_cells['imaging_modality']=="EM",['shifted_best_match_function',
                                                                     'shifted_best_match_morphology',
                                                                     'cell_name',
                                                                     'shifted_best_match_value',
                                                                     'shifted_best_match']].groupby(['shifted_best_match_function',
                                                                                                 'shifted_best_match_morphology',
                                                                                                 'cell_name']).max()
    grouped_shifted = grouped_shifted.sort_values(['shifted_best_match_function','shifted_best_match_morphology'],ascending=False)



    nb_avg = (np.array(nb_df) + np.array(nb2_df))/2
    nb_avg_df = pd.DataFrame(nb_avg)
    nb_avg_df.columns = nb_df.columns
    nb_avg_df.index = nb_df.index

    for i,cell in nb_avg_df.iterrows():
        all_cells.loc[all_cells['cell_name']==i,"avg_best_match"] = cell.index[cell.argmax()]
        all_cells.loc[all_cells['cell_name'] == i, "avg_best_match_value"] = cell.max()
        all_cells.loc[all_cells['cell_name'] == i, "avg_best_match_function"] = all_cells.loc[all_cells['cell_name'] == cell.index[cell.argmax()], "function"].iloc[0]
        all_cells.loc[all_cells['cell_name'] == i, "avg_best_match_morphology"] = all_cells.loc[all_cells['cell_name'] == cell.index[cell.argmax()], "morphology"].iloc[0]
        all_cells.loc[all_cells['cell_name'] == i, "avg_best_match_neurotransmitter"] = all_cells.loc[all_cells['cell_name'] == cell.index[cell.argmax()], "neurotransmitter"].iloc[0]

    grouped_avg = all_cells.loc[all_cells['imaging_modality']=="EM",['avg_best_match_function',
                                                                     'avg_best_match_morphology',
                                                                     'cell_name',
                                                                     'avg_best_match_value',
                                                                     'avg_best_match']].groupby(['avg_best_match_function',
                                                                                                 'avg_best_match_morphology',
                                                                                                 'cell_name']).max()
    grouped_avg = grouped_avg.sort_values(['avg_best_match_function','avg_best_match_morphology'],ascending=False)

