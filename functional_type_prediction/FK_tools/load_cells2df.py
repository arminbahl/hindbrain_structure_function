import numpy as np
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.visualization.FK_tools.load_em_table import *
from hindbrain_structure_function.visualization.FK_tools.load_mesh import *
from hindbrain_structure_function.visualization.FK_tools.load_brs import *
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
from pathlib import Path


def load_cells_predictor_pipeline(modalities=['pa','clem','em'],
                                  mirror = True,
                                  keywords = 'all',
                                  path_to_data=Path(r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data"),
                                  use_smooth=True,
                                  load_repaired=False,
                                  load_both=False,
                                  summarize_off_response=True, input_em=True):
    # Load the photoactivation table if 'pa' modality is selected; path assumes a specific directory structure.
    table_list = []
    if 'pa' in modalities:

        pa_table = load_pa_table(path_to_data.joinpath("paGFP").joinpath("photoactivation_cells_table.csv"))
        table_list.append(pa_table)

    # Load the CLEM table if 'clem' modality is selected; path also assumes a specific directory structure.
    if 'clem' in modalities:
        clem_table = load_clem_table(path_to_data.joinpath('clem_zfish1').joinpath('functionally_imaged'))
        table_list.append(clem_table)
    if 'clem_predict' in modalities:
        clem_predict_table = load_clem_table(
            path_to_data.joinpath('clem_zfish1').joinpath('non_functionally_imaged'))
        table_list.append(clem_predict_table)


        table_list.append(clem_predict_table)
    if 'clem241211' in modalities:
        clem241211_table = load_clem_table(
            path_to_data.joinpath('clem_zfish1').joinpath('new_batch_111224').joinpath('functionally_imaged_111224'))
        table_list.append(clem241211_table)

    if 'clem_predict241211' in modalities:
        clem_predict241211_table = load_clem_table(path_to_data.joinpath('clem_zfish1').joinpath('new_batch_111224').joinpath(
            'non_functionally_imaged_111224'))
        table_list.append(clem_predict241211_table)


    if 'em' in modalities:
        em_table1 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('data_cell_89189_postsynaptic_partners').joinpath('output_data'),'89189')
        em_table2 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('data_seed_cells').joinpath('output_data'),'seed_cell')
        em_table3 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('cell_010_postsynaptic_partners').joinpath('output_data'),'13772')
        em_table4 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('cell_011_postsynaptic_partners').joinpath('output_data'),'149747')
        em_table5 = load_em_table(
            path_to_data.joinpath('em_zfish1').joinpath('cell_003_postsynaptic_partners').joinpath('output_data'),
            'aaaaa')
        em_table6 = load_em_table(
            path_to_data.joinpath('em_zfish1').joinpath('cell_019_postsynaptic_partners').joinpath('output_data'),
            '119243')
        em_table_dt = load_em_table(
            path_to_data.joinpath('em_zfish1').joinpath('search4putativeDTs').joinpath('output_data'), 'DT')
        if input_em:
            em_table7 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('presynapses').joinpath(
                'cell_89189_presynaptic_partners').joinpath('output_data'), 'p89189')
            em_table8 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('presynapses').joinpath(
                'cell_119243_presynaptic_partners').joinpath('output_data'), 'p119243')
            em_table9 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('presynapses').joinpath(
                'cell_137722_presynaptic_partners').joinpath('output_data'), 'p137722')
            em_table10 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('presynapses').joinpath(
                'cell_149747_presynaptic_partners').joinpath('output_data'), 'p149747')
            em_table11 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('presynapses').joinpath(
                'cell_147009_presynaptic_partners').joinpath('output_data'), 'p147009')

            em_table = pd.concat(
                [em_table1, em_table2, em_table3, em_table4, em_table5, em_table6, em_table7, em_table8, em_table9,
                 em_table10, em_table11,
                 em_table_dt])
            em_table.loc[:, "classifier"] = None
            table_list.append(em_table)
        else:

            em_table = pd.concat([em_table1, em_table2, em_table3, em_table4, em_table5, em_table_dt])
            em_table.loc[:, "classifier"] = None
            table_list.append(em_table)




    # Concatenate data from different modalities into a single DataFrame if multiple modalities are specified.
    if len(modalities) > 1:
        all_cells = pd.concat(table_list)
    elif len(modalities) == 1:
        all_cells = eval(modalities[0] + "_table")
    all_cells = all_cells.reset_index(drop=True)

    keywords = keywords
    # Filter the concatenated cell data based on specified keywords.
    if keywords != 'all':
        for keyword in keywords:
            subset_for_keyword = all_cells['cell_type_labels'].apply(lambda label: keyword.replace("_", " ") in label or keyword in label)
            all_cells = all_cells[subset_for_keyword]


    # Initialize columns for different types of mesh data, setting default as NaN.
    for mesh_type in ['soma_mesh', 'dendrite_mesh', 'axon_mesh', 'neurites_mesh', 'swc','all_mesh']:
        all_cells[mesh_type] = np.nan
        all_cells[mesh_type] = all_cells[mesh_type].astype(object)

    # If a cell was scored by Jonathan Boulanger-Weill, set its imaging modality to 'clem'.
    try:
        all_cells.loc[all_cells['tracer_names'] == 'Jonathan Boulanger-Weill', 'imaging_modality'] = 'clem'  # Confirm with Jonathan regarding the use of 'clem' as a label.
        all_cells.loc[all_cells['tracer_names'] == 'Jay Hareshbhai Savaliya', 'imaging_modality'] = 'clem'
        all_cells.loc[all_cells['functional_id'] == 'not functionally imaged', 'imaging_modality'] = 'clem'
    except:
        pass


    # Load mesh data for each cell based on selected modalities and smoothing setting.
    for i, cell in all_cells.iterrows():
        # all_cells.loc[i, :] = load_mesh(cell, path_to_data, use_smooth_pa=use_smooth, swc=True,load_repaired=load_repaired,load_both=load_both)
        all_cells.loc[i, :] = load_mesh_new(cell, use_smooth_pa=use_smooth, swc=True, load_repaired=load_repaired,
                                            load_both=load_both)
        if type(all_cells.loc[i,'swc']) == float:
            print(f'{cell.cell_name} is not a TreeNeuron\n')



    # Mirror cell data if specified, adjusting for anatomical accuracy.
    width_brain = 495.56  # The width of the brain for mirror transformations.
    if mirror:
        for i, cell in all_cells.iterrows():
            if cell.cell_name == "cell_576460752665417287":
                pass
            if type(cell['soma_mesh']) != float and type(cell['soma_mesh']) != type(None):
                if np.mean(cell['soma_mesh']._vertices[:, 0]) > (width_brain / 2):  # Determine if the cell is in the right hemisphere.
                    # Mirror various mesh data based on imaging modality.
                    all_cells.loc[i, 'soma_mesh']._vertices = navis.transforms.mirror(cell['soma_mesh']._vertices, width_brain, 'x')
                    if cell['imaging_modality'] == 'photoactivation':
                        all_cells.loc[i, 'neurites_mesh']._vertices = navis.transforms.mirror(cell['neurites_mesh']._vertices, width_brain, 'x')
                        all_cells.loc[i, 'all_mesh']._vertices = navis.transforms.mirror(cell['all_mesh']._vertices, width_brain, 'x')
                    if cell['imaging_modality'] == 'clem' or cell['imaging_modality'] == 'em':
                        all_cells.loc[i, 'axon_mesh']._vertices = navis.transforms.mirror(cell['axon_mesh']._vertices, width_brain, 'x')
                        all_cells.loc[i, 'dendrite_mesh']._vertices = navis.transforms.mirror(cell['dendrite_mesh']._vertices, width_brain, 'x')
                        if type(cell['all_mesh'].connectors) != float and cell['all_mesh'].connectors is not None:
                            all_cells.loc[i, 'all_mesh'].connectors.loc[:, ["x", 'y', 'z']] = navis.transforms.mirror(np.array(cell['all_mesh'].connectors.loc[:, ['x', 'y', 'z']]), width_brain, 'x')
                    print(f"MESHES of cell {cell['cell_name']} mirrored")

            if 'swc' in cell.index:
                if type(cell['swc']) != float and type(cell['swc']) != type(None):
                    if cell['swc'].nodes.loc[0, 'x'] > (width_brain / 2):
                        all_cells.loc[i, 'swc'].nodes.loc[:, ["x", "y", "z"]] = navis.transforms.mirror(np.array(cell['swc'].nodes.loc[:, ['x', 'y', 'z']]), width_brain, 'x')
                        if all_cells.loc[i, 'swc'].connectors is not None:
                            all_cells.loc[i, 'swc'].connectors.loc[:, ["x", 'y', 'z']] = navis.transforms.mirror(np.array(cell['swc'].connectors.loc[:, ['x', 'y', 'z']]), width_brain, 'x')
                        print(f"SWC of cell {cell['cell_name']} mirrored")

    #set some correct datatypes
    all_cells = all_cells.dropna(how='all')
    print(f"{np.sum([type(x)==float for x in all_cells['swc']])} cells dropped because no swc")
    print(all_cells.loc[[type(x) != navis.TreeNeuron for x in all_cells['swc']],'cell_name'])
    all_cells = all_cells.loc[[type(x)!=float for x in all_cells['swc']],:]



    #extract features from pa cell labels
    neurotransmitter_dict = {'gad1b':'inhibitory','gad1':'inhibitory','vglut':'excitatory','vglut2':'excitatory','vglut2a':'excitatory'}
    cell_type_categories = {'morphology': ['ipsilateral', 'contralateral'],
                            'neurotransmitter': ['inhibitory', 'excitatory'],
                            'function': ['integrator', 'dynamic_threshold', 'dynamic threshold', 'motor_command', 'motor command','no response','off-response','motion responsive',"noisy, little modulation","non-direction-selective, on response"]}
    for i, cell in all_cells.iterrows():
        if cell.imaging_modality != "EM":
            if type(cell.cell_type_labels) == list:
                for label in cell.cell_type_labels:
                    if label in cell_type_categories['morphology']:
                        all_cells.loc[i, 'morphology'] = label
                    elif label in cell_type_categories['function']:
                        all_cells.loc[i, 'function'] = label
                        if label == 'nan':
                            pass
                    elif label in cell_type_categories['neurotransmitter']:
                        all_cells.loc[i, 'neurotransmitter'] = label
                try:
                    all_cells.loc[i, 'swc'].name = all_cells.loc[i, 'swc'].name + " NT: " + all_cells.loc[i, 'neurotransmitter']
                except:
                    pass


        if cell.imaging_modality == "EM":

            if True in list(cell['swc'].nodes.loc[:, "x"] > (width_brain / 2)+10):
                all_cells.loc[i, 'morphology'] = 'contralateral'
            else:
                all_cells.loc[i, 'morphology'] = 'ipsilateral'
            try:
                if cell.cell_name == "147009" or cell.cell_name == "102596":
                    all_cells.loc[i, 'neurotransmitter'] = 'inhibitory'
                    pass
                else:
                    path_to_neurotransmitter_xlsx = \
                        [x for x in cell.metadata_path.parent.parent.parent.glob('*.xlsx') if "inh_exc" in str(x)][0]
                    neurotransmitter_df = pd.read_excel(path_to_neurotransmitter_xlsx)
                    if not neurotransmitter_df.loc[
                        neurotransmitter_df['id'] == int(cell.cell_name), 'neurotransmitter_ID'].empty:
                        try:
                            neurotransmitter = neurotransmitter_df.loc[
                                neurotransmitter_df['id'] == int(cell.cell_name), 'neurotransmitter_ID'].item().lower()
                        except:
                            neurotransmitter = neurotransmitter_df.loc[
                                neurotransmitter_df['id'] == int(cell.cell_name), 'neurotransmitter_ID'].item()
                    else:
                        neurotransmitter = 'nan'
                    if neurotransmitter in neurotransmitter_dict.keys():
                        all_cells.loc[i, 'neurotransmitter'] = neurotransmitter_dict[neurotransmitter]
                    else:
                        all_cells.loc[i, 'neurotransmitter'] = 'na'
            except:
                all_cells.loc[i, 'neurotransmitter'] = 'na'


    #load neurotransmitter into gregors EM cells





    # #sort dfs
    # all_cells_pa = all_cells.sort_values(['function','morphology','neurotransmitter'])
    # all_cells_em = all_cells.sort_values('classifier')


    # Finalize the all_cells attribute with the loaded and possibly transformed cell data.
    all_cells = all_cells.dropna(how='all')

    if not 'function' in all_cells.columns:
        all_cells.loc[:, 'function'] = np.nan
    if not 'reconstruction_status' in all_cells.columns:
        all_cells.loc[:, 'reconstruction_status'] = np.nan

    all_cells.loc[all_cells['function'].isna(), 'function'] = 'to_predict'
    all_cells['function'] = all_cells['function'].apply(lambda x: x.replace(" ","_"))
    if summarize_off_response:
        all_cells.loc[(~all_cells['function'].isin(['integrator','dynamic_threshold','motor_command']))&
                      (all_cells['function']!='to_predict'), 'function'] = 'neg_control'

    temp_bool0 = all_cells['imaging_modality'] == 'clem'
    temp_bool1 = all_cells['reconstruction_status'] != 'axon complete, dendrites complete'
    temp_bool2 = all_cells['function'].isin(['integrator', 'dynamic_threshold', 'motor_command'])
    temp_bool_all = np.array(temp_bool0 * temp_bool1 * temp_bool2)

    all_cells = all_cells.loc[~(temp_bool_all), :]
    return all_cells

if __name__ == '__main__':
   all_cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'),modalities=['clem_predict'],load_repaired=True)

