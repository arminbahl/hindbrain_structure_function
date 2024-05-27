from hindbrain_structure_function.visualization.FK_tools.load_pa_table import *
from hindbrain_structure_function.visualization.FK_tools.load_clem_table import *
from hindbrain_structure_function.visualization.FK_tools.load_em_table import *
from hindbrain_structure_function.visualization.FK_tools.load_mesh import *
from hindbrain_structure_function.visualization.FK_tools.load_brs import *
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
from pathlib import Path


def load_cells_predictor_pipeline(modalities=['pa','clem','em'],
                                  mirror = True,
                                  keywords = 'all',
                                  path_to_data=Path(r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data")):
    # Load the photoactivation table if 'pa' modality is selected; path assumes a specific directory structure.
    table_list = []
    if 'pa' in modalities:

        pa_table = load_pa_table(path_to_data.joinpath("paGFP").joinpath("photoactivation_cells_table.csv"))
        table_list.append(pa_table)

    # Load the CLEM table if 'clem' modality is selected; path also assumes a specific directory structure.
    if 'clem' in modalities:
        clem_table = load_clem_table(path_to_data.joinpath('clem_zfish1').joinpath('all_cells'))
        table_list.append(clem_table)

    if 'em' in modalities:
        em_table1 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('data_cell_89189_postsynaptic_partners').joinpath('output_data'))
        em_table2 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('data_seed_cells').joinpath('output_data'))
        em_table3 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('cell_010_postsynaptic_partners').joinpath('output_data'))
        em_table4 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('cell_011_postsynaptic_partners').joinpath('output_data'))
        em_table5 = load_em_table(path_to_data.joinpath('em_zfish1').joinpath('cell_019_postsynaptic_partners').joinpath('output_data'))
        em_table = pd.concat([em_table1, em_table2,em_table3,em_table4,em_table5])
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
    for mesh_type in ['soma_mesh', 'dendrite_mesh', 'axon_mesh', 'neurites_mesh', 'swc']:
        all_cells[mesh_type] = np.nan
        all_cells[mesh_type] = all_cells[mesh_type].astype(object)

    # If a cell was scored by Jonathan Boulanger-Weill, set its imaging modality to 'clem'.
    try:
        all_cells.loc[all_cells['tracer_names'] == 'Jonathan Boulanger-Weill', 'imaging_modality'] = 'clem'  # Confirm with Jonathan regarding the use of 'clem' as a label.
        all_cells.loc[all_cells['functional_id'] == 'not functionally imaged', 'imaging_modality'] = 'clem'
    except:
        pass


    # Load mesh data for each cell based on selected modalities and smoothing setting.
    for i, cell in all_cells.iterrows():
        all_cells.loc[i, :] = load_mesh(cell, path_to_data, use_smooth_pa=True, swc=True)
        if type(all_cells.loc[i,'swc']) == float:
            print(f'{cell.cell_name} is not a TreeNeuron\n')

    # Mirror cell data if specified, adjusting for anatomical accuracy.
    width_brain = 495.56  # The width of the brain for mirror transformations.
    if mirror:
        for i, cell in all_cells.iterrows():
            if type(cell['soma_mesh']) != float and type(cell['soma_mesh']) != type(None):
                if np.mean(cell['soma_mesh']._vertices[:, 0]) > (width_brain / 2):  # Determine if the cell is in the right hemisphere.
                    # Mirror various mesh data based on imaging modality.
                    all_cells.loc[i, 'soma_mesh']._vertices = navis.transforms.mirror(cell['soma_mesh']._vertices, width_brain, 'x')
                    if cell['imaging_modality'] == 'photoactivation':
                        all_cells.loc[i, 'neurites_mesh']._vertices = navis.transforms.mirror(cell['neurites_mesh']._vertices, width_brain, 'x')
                    if cell['imaging_modality'] == 'clem' or cell['imaging_modality'] == 'em':
                        all_cells.loc[i, 'axon_mesh']._vertices = navis.transforms.mirror(cell['axon_mesh']._vertices, width_brain, 'x')
                        all_cells.loc[i, 'dendrite_mesh']._vertices = navis.transforms.mirror(cell['dendrite_mesh']._vertices, width_brain, 'x')

            if type(cell['swc']) != float and type(cell['swc']) != type(None):
                if cell['swc'].nodes.loc[0,'x'] > (width_brain / 2):
                    all_cells.loc[i,'swc'].nodes.loc[:,["x","y","z"]] = navis.transforms.mirror(np.array(cell['swc'].nodes.loc[:,['x','y','z']]), width_brain, 'x')


    # Finalize the all_cells attribute with the loaded and possibly transformed cell data.
    all_cells = all_cells.dropna(how='all')
    all_cells = all_cells
    return all_cells

if __name__ == '__main__':
    all_cells = load_cells_predictor_pipeline()

