from hindbrain_structure_function.functional_type_prediction.FK_tools.find_branches import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.fragment_neurite import *
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def calculate_metric2df(cell_df, file_name, path_to_data, force_new=False):
    def check_skip_condition(path_to_data, file_name, key, cell_df, force_new):
        file_path = path_to_data / 'prediction' / 'features' / f'{file_name}_features.hdf5'
        if file_path.exists():
            with h5py.File(file_path, 'r') as h5file:
                if key in h5file.keys() and h5file['angle_cross/axis1'].shape[0] == cell_df.shape[0] and not force_new:
                    return True
        return False
    def ic_index(x_coords):
        width_brain = 495.56

        distances = []
        for x in x_coords:
            distances.append(((width_brain / 2) - x) / (width_brain / 2))
        ipsi_contra_index = np.sum(distances) / len(distances)
        return ipsi_contra_index

    width_brain = 495.56


    if not check_skip_condition(path_to_data, file_name,  'predictor_pipeline_features', cell_df, force_new):
        # resample
        cell_df.loc[:, 'not_resampled_swc'] = cell_df['swc']
        cell_df.loc[:, 'swc'] = cell_df.swc.apply(lambda x: x.resample("0.5 micron"))

        # reset index
        cell_df = cell_df.reset_index(drop=True)

        # predictor pipeline
        cell_df.loc[:, "contralateral_branches"] = cell_df.loc[:, "swc"].apply(lambda x: len(x.nodes.loc[(x.nodes.x > width_brain / 2) & (x.nodes.type == 'branch'), 'type']))
        cell_df.loc[:, "ipsilateral_branches"] = cell_df.loc[:, "swc"].apply(lambda x: len(x.nodes.loc[(x.nodes.x < width_brain / 2) & (x.nodes.type == 'branch'), 'type']))

        # add cable length
        cell_df.loc[:, 'cable_length'] = cell_df.loc[:, "swc"].apply(lambda x: x.cable_length)
        # add bbox volume
        cell_df.loc[:, 'bbox_volume'] = cell_df.loc[:, "swc"].apply(lambda x: (x.extents[0]) * (x.extents[1]) * (x.extents[2]))
        # add x_extenct
        cell_df.loc[:, 'x_extent'] = cell_df.loc[:, "swc"].apply(lambda x: x.extents[0])
        # add y_extenct
        cell_df.loc[:, 'y_extent'] = cell_df.loc[:, "swc"].apply(lambda x: x.extents[1])
        # add z_extenct
        cell_df.loc[:, 'z_extent'] = cell_df.loc[:, "swc"].apply(lambda x: x.extents[2])

        # add avg x,y,z coordinate
        cell_df.loc[:, 'x_avg'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.x))
        cell_df.loc[:, 'y_avg'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.y))
        cell_df.loc[:, 'z_avg'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.z))

        # add soma x,y,z coordinate
        cell_df.loc[:, 'soma_x'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.loc[0, "x"]))
        cell_df.loc[:, 'soma_y'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.loc[0, "y"]))
        cell_df.loc[:, 'soma_z'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.loc[0, "z"]))

        #
        cell_df.loc[:, 'tortuosity'] = cell_df.loc[:, "swc"].apply(lambda x: navis.tortuosity(x))

        # add n_leafs
        cell_df.loc[:, 'n_leafs'] = cell_df.loc[:, "swc"].apply(lambda x: x.n_leafs)
        # add n_branches
        cell_df.loc[:, 'n_branches'] = cell_df.loc[:, "swc"].apply(lambda x: x.n_branches)
        # add n_ends
        cell_df.loc[:, "n_ends"] = cell_df.loc[:, "swc"].apply(lambda x: x.n_ends)
        # add n_edges
        cell_df.loc[:, "n_edges"] = cell_df.loc[:, "swc"].apply(lambda x: x.n_edges)
        # main brainchpoint
        cell_df.loc[:, "main_branchpoint"] = cell_df.loc[:, "swc"].apply(lambda x: navis.find_main_branchpoint(x))

        # number of persitence points n_persistence_points
        cell_df.loc[:, "n_persistence_points"] = cell_df.loc[:, "swc"].apply(lambda x: len(navis.persistence_points(x)))
        # add strahler index
        _ = cell_df.loc[:, "swc"].apply(lambda x: navis.strahler_index(x))
        # add max strahler index
        cell_df.loc[:, "max_strahler_index"] = cell_df.loc[:, "swc"].apply(lambda x: x.nodes.strahler_index.max())

        # add sholl distance most branches
        sholl_analysis_results = cell_df.loc[:, "swc"].apply(lambda x: navis.sholl_analysis(x, radii=np.arange(10, 200, 10), center='root'))
        cell_df.loc[:, "sholl_distance_max_branches"] = sholl_analysis_results.apply(lambda x: x.branch_points.idxmax())
        cell_df.loc[:, "sholl_distance_max_branches_cable_length"] = sholl_analysis_results.apply(lambda x: x.cable_length[x.branch_points.idxmax()])
        cell_df.loc[:, "sholl_distance_max_branches_geosidic"] = sholl_analysis_results.apply(lambda x: x.branch_points.idxmax())
        cell_df.loc[:, "sholl_distance_max_branches_geosidic_cable_length"] = sholl_analysis_results.apply(lambda x: x.cable_length[x.branch_points.idxmax()])
        branches_df = None
        for i, cell in tqdm(cell_df.iterrows(), leave=False, total=len(cell_df)):
            temp = find_branches(cell['swc'].nodes, cell.cell_name)
            if type(branches_df) == type(None):
                branches_df = temp
            else:
                branches_df = pd.concat([branches_df, temp])

        for i, cell in tqdm(cell_df.iterrows()):
            cell_df.loc[i, "main_path_longest_neurite"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name) &
                                                                          (branches_df['main_path']) &
                                                                          (branches_df['end_type'] != 'end'), 'longest_neurite_in_branch'].iloc[0]
            cell_df.loc[i, "main_path_total_branch_length"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name) &
                                                                              (branches_df['main_path']) &
                                                                              (branches_df['end_type'] != 'end'), 'total_branch_length'].iloc[0]

            try:
                cell_df.loc[i, "first_major_branch_longest_neurite"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name) &
                                                                                       (~branches_df['main_path']) &
                                                                                       (branches_df['end_type'] != 'end') &
                                                                                       (branches_df['total_branch_length'] >= 50), 'longest_neurite_in_branch'].iloc[0]
            except:
                cell_df.loc[i, "first_major_branch_longest_neurite"] = 0
            try:
                cell_df.loc[i, "first_major_branch_total_branch_length"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name) &
                                                                                           (~branches_df['main_path']) &
                                                                                           (branches_df['end_type'] != 'end') &
                                                                                           (branches_df['total_branch_length'] >= 50), 'total_branch_length'].iloc[0]
            except:
                cell_df.loc[i, "first_major_branch_total_branch_length"] = 0

            fragmented_neuron = navis.split_into_fragments(cell_df.loc[i, "swc"], cell_df.loc[i, "swc"].n_leafs)
            cell_df.loc[i, "first_branch_longest_neurite"] = navis.longest_neurite(fragmented_neuron[1]).cable_length
            cell_df.loc[i, "first_branch_total_branch_length"] = fragmented_neuron[1].cable_length

            temp = cell_df.loc[i, "swc"]
            temp = navis.prune_twigs(temp, 5, recursive=True)
            temp_node_id = temp.nodes.loc[temp.nodes.type == 'branch', 'node_id'].iloc[0]
            temp = navis.cut_skeleton(temp, temp_node_id)
            cell_df.loc[i, "cable_length_2_first_branch"] = temp[1].cable_length
            cell_df.loc[i, "z_distance_first_2_first_branch"] = temp[1].nodes.iloc[0].z - temp[1].nodes.iloc[-1].z

            # biggest major branch
            cell_df.loc[i, "biggest_branch_longest_neurite"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name) &
                                                                               (~branches_df['main_path']) &
                                                                               (branches_df['end_type'] != 'end'), :].sort_values('total_branch_length', ascending=False)[
                'longest_neurite_in_branch'].iloc[0]
            cell_df.loc[i, "biggest_branch_total_branch_length"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name) &
                                                                                   (~branches_df['main_path']) &
                                                                                   (branches_df['end_type'] != 'end'), 'total_branch_length'].iloc[0]

            cell_df.loc[i, "longest_connected_path"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name), 'longest_connected_path'].iloc[0]

            cell_df.loc[i, 'n_nodes_ipsi_hemisphere'] = (cell.swc.nodes.x < (width_brain / 2)).sum()
            cell_df.loc[i, 'n_nodes_contra_hemisphere'] = (cell.swc.nodes.x > (width_brain / 2)).sum()

            cell_df.loc[i, 'n_nodes_ipsi_hemisphere_fraction'] = ((cell.swc.nodes.x < (width_brain / 2)).sum()) / len(cell.swc.nodes.x)
            cell_df.loc[i, 'n_nodes_contra_hemisphere_fraction'] = ((cell.swc.nodes.x > (width_brain / 2)).sum()) / len(cell.swc.nodes.x)



            cell_df.loc[i, 'x_location_index'] = ic_index(cell.swc.nodes.x)

            cell_df.loc[i, 'fraction_contra'] = (cell.swc.nodes.x > (width_brain / 2)).sum() / len(cell.swc.nodes.x)
            cell_df.loc[i, 'y_extent_ipsi'] = cell.swc.nodes.loc[cell.swc.nodes.x < (width_brain / 2), "y"].max() - cell.swc.nodes.loc[cell.swc.nodes.x < (width_brain / 2), "y"].min()
            cell_df.loc[i, 'z_extent_ipsi'] = cell.swc.nodes.loc[cell.swc.nodes.x < (width_brain / 2), "z"].max() - cell.swc.nodes.loc[cell.swc.nodes.x < (width_brain / 2), "z"].min()

            cell_df.loc[i, 'max_x_ipsi'] = cell.swc.nodes.loc[cell.swc.nodes.x < (width_brain / 2), "x"].max()
            cell_df.loc[i, 'max_y_ipsi'] = cell.swc.nodes.loc[cell.swc.nodes.x < (width_brain / 2), "y"].max()
            cell_df.loc[i, 'max_z_ipsi'] = cell.swc.nodes.loc[cell.swc.nodes.x < (width_brain / 2), "z"].max()

            cell_df.loc[i, 'min_x_ipsi'] = cell.swc.nodes.loc[cell.swc.nodes.x < (width_brain / 2), "x"].min()
            cell_df.loc[i, 'min_y_ipsi'] = cell.swc.nodes.loc[cell.swc.nodes.x < (width_brain / 2), "y"].min()
            cell_df.loc[i, 'min_z_ipsi'] = cell.swc.nodes.loc[cell.swc.nodes.x < (width_brain / 2), "z"].min()

            cell_df.loc[i, 'max_x_contra'] = cell.swc.nodes.loc[cell.swc.nodes.x > (width_brain / 2), "x"].max()
            cell_df.loc[i, 'max_y_contra'] = cell.swc.nodes.loc[cell.swc.nodes.x > (width_brain / 2), "y"].max()
            cell_df.loc[i, 'max_z_contra'] = cell.swc.nodes.loc[cell.swc.nodes.x > (width_brain / 2), "z"].max()

            cell_df.loc[i, 'min_x_contra'] = cell.swc.nodes.loc[cell.swc.nodes.x > (width_brain / 2), "x"].min()
            cell_df.loc[i, 'min_y_contra'] = cell.swc.nodes.loc[cell.swc.nodes.x > (width_brain / 2), "y"].min()
            cell_df.loc[i, 'min_z_contra'] = cell.swc.nodes.loc[cell.swc.nodes.x > (width_brain / 2), "z"].min()
            if np.isnan(cell_df.loc[i, 'max_x_contra']):
                cell_df.loc[i, 'max_x_contra'] = 0
                cell_df.loc[i, 'max_y_contra'] = 0
                cell_df.loc[i, 'max_z_contra'] = 0
                cell_df.loc[i, 'min_x_contra'] = 0
                cell_df.loc[i, 'min_y_contra'] = 0
                cell_df.loc[i, 'min_z_contra'] = 0
            cell_df.loc[i, 'avg_delta_death_birth_persitence'] = np.mean([row['death'] - row['birth'] for i, row in navis.persistence_points(cell.swc).iterrows()])
            cell_df.loc[i, 'median_delta_death_birth_persitence'] = np.mean([row['death'] - row['birth'] for i, row in navis.persistence_points(cell.swc).iterrows()])
            cell_df.loc[i, 'std_delta_death_birth_persitence'] = np.std([row['death'] - row['birth'] for i, row in navis.persistence_points(cell.swc).iterrows()])

            cell_df.loc[i, 'z_extent_contra'] = cell.swc.nodes.loc[cell.swc.nodes.x > (width_brain / 2), "z"].max() - cell.swc.nodes.loc[cell.swc.nodes.x > (width_brain / 2), "z"].min()
            cell_df.loc[i, 'y_extent_contra'] = cell.swc.nodes.loc[cell.swc.nodes.x > (width_brain / 2), "y"].max() - cell.swc.nodes.loc[cell.swc.nodes.x > (width_brain / 2), "y"].min()
            if np.isnan(cell_df.loc[i, 'y_extent_contra']):
                cell_df.loc[i, 'z_extent_contra'] = 0
                cell_df.loc[i, 'y_extent_contra'] = 0

        temp1_index = list(cell_df.columns).index('contralateral_branches')
        temp1 = cell_df.loc[:, cell_df.columns[temp1_index:]]

        temp1.to_hdf(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5', 'predictor_pipeline_features')

        temp2 = cell_df.loc[:, ['cell_name', 'imaging_modality', 'function', 'morphology', 'neurotransmitter']]
        temp2.to_hdf(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5', 'function_morphology_neurotransmitter')



    if not check_skip_condition(path_to_data, file_name, 'angle_cross', cell_df, force_new):
        # extract branching angle and coords of crossing for contralateral neurons
        for i, cell in tqdm(cell_df.iterrows(), total=cell_df.shape[0], leave=False):
            if cell.morphology == 'contralateral':
                # pass

                angle, crossing_coords, fragments_list = direct_angle_and_crossing_extraction(cell['not_resampled_swc'].nodes, projection="3d")
                angle2d, crossing_coords, fragments_list = direct_angle_and_crossing_extraction(cell['not_resampled_swc'].nodes, projection="2d")

                if np.isnan(angle):
                    pass
                try:
                    cell_df.loc[i, 'angle'] = angle
                    cell_df.loc[i, 'angle2d'] = angle2d
                    cell_df.loc[i, 'x_cross'] = crossing_coords[0]
                    cell_df.loc[i, 'y_cross'] = crossing_coords[1]
                    cell_df.loc[i, 'z_cross'] = crossing_coords[2]

                except:
                    cell_df.loc[i, 'angle'] = np.nan
                    cell_df.loc[i, 'angle2d'] = np.nan
                    cell_df.loc[i, 'x_cross'] = np.nan
                    cell_df.loc[i, 'y_cross'] = np.nan
                    cell_df.loc[i, 'z_cross'] = np.nan
            else:
                cell_df.loc[i, 'angle'] = np.nan
                cell_df.loc[i, 'angle2d'] = np.nan
                cell_df.loc[i, 'x_cross'] = np.nan
                cell_df.loc[i, 'y_cross'] = np.nan
                cell_df.loc[i, 'z_cross'] = np.nan

        temp3 = cell_df.loc[:, ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross']]
        temp3.to_hdf(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5', 'angle_cross')

        complete_df = pd.concat([temp2, temp1, temp3], axis=1)
        complete_df.to_hdf(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5', 'complete_df')




def load_train_data_df(path, file="CLEM_and_PA"):
    # file_path = path / 'make_figures_FK_output' / 'prediction_project_features.hdf5'

    file_path = path / 'make_figures_FK_output' / f'{file}_train_features.hdf5'
    all_cells = pd.read_hdf(file_path, 'complete_df')

    # throw out weird jon cells

    # Data Preprocessing
    without_nan_function = all_cells[all_cells['function'] != 'nan']
    without_nan_function = without_nan_function.sort_values(by=['function', 'morphology', 'imaging_modality', 'neurotransmitter'])
    # Impute NaNs
    columns_possible_nans = ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross']
    without_nan_function.loc[:, columns_possible_nans] = without_nan_function[columns_possible_nans].fillna(0)

    # Function string replacement
    without_nan_function.loc[:, 'function'] = without_nan_function['function'].str.replace('_', ' ')

    # Update 'integrator' function
    def update_integrator(df):
        integrator_mask = df['function'] == 'integrator'
        df.loc[integrator_mask, 'function'] += " " + df.loc[integrator_mask, 'morphology']

    update_integrator(without_nan_function)

    # Replace strings with indices
    columns_replace_string = ['neurotransmitter', 'morphology']
    neurotransmitter2int_dict = {'excitatory': 0, 'inhibitory': 1, 'na': 2}
    morphology2int_dict = {'contralateral': 0, 'ipsilateral': 1}

    for work_column in columns_replace_string:
        without_nan_function.loc[:, work_column + "_clone"] = without_nan_function[work_column]
        for key in eval(f'{work_column}2int_dict').keys():
            without_nan_function.loc[without_nan_function[work_column] == key, work_column] = eval(f'{work_column}2int_dict')[key]

    # Standardize features
    scaler = StandardScaler()

    features_df1 = without_nan_function.drop(['cell_name', 'imaging_modality', 'function', "neurotransmitter_clone", 'morphology_clone'], axis=1)
    non_features_df = without_nan_function.loc[:, ['cell_name', 'imaging_modality', 'function', "neurotransmitter_clone", 'morphology_clone']]
    features = scaler.fit_transform(features_df1.to_numpy())
    features_df = pd.DataFrame(features, columns=features_df1.columns)
    finished_df = pd.concat([non_features_df, features_df], axis=1)
    # remove weird jon cells?
    finished_df = finished_df.loc[~finished_df.cell_name.isin(["cell_576460752734566521", "cell_576460752723528109", "cell_576460752684182585"]), :]

    return finished_df


def load_predict_data_df(path, file):
    file_path = path / 'make_figures_FK_output' / f'{file}_predict_features.hdf5'
    all_cells = pd.read_hdf(file_path, 'complete_df')

    # throw out weird jon cells
    all_cells = all_cells.loc[~all_cells.cell_name.isin(["cell_576460752734566521", "cell_576460752723528109", "cell_576460752684182585"]), :]
    cell_names = all_cells.cell_name
    # Data Preprocessing
    without_nan_function = all_cells
    without_nan_function = without_nan_function.sort_values(by=['morphology', 'imaging_modality', 'neurotransmitter'])

    # Impute NaNs
    columns_possible_nans = ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross']
    without_nan_function.loc[:, columns_possible_nans] = without_nan_function[columns_possible_nans].fillna(0)

    # Replace strings with indices
    columns_replace_string = ['neurotransmitter', 'morphology']
    neurotransmitter2int_dict = {'excitatory': 0, 'inhibitory': 1, 'na': 2}
    morphology2int_dict = {'contralateral': 0, 'ipsilateral': 1}

    for work_column in columns_replace_string:
        without_nan_function.loc[:, work_column + "_clone"] = without_nan_function[work_column]
        for key in eval(f'{work_column}2int_dict').keys():
            without_nan_function.loc[without_nan_function[work_column] == key, work_column] = eval(f'{work_column}2int_dict')[key]

    # sort by function an imaging modality

    # Standardize features
    scaler = StandardScaler()
    try:
        features_df1 = without_nan_function.drop(['cell_name', 'imaging_modality', 'function', "neurotransmitter_clone", 'morphology_clone'], axis=1)
        non_features_df = without_nan_function.loc[:, ['cell_name', 'imaging_modality', 'function', "neurotransmitter_clone", 'morphology_clone']]
    except:
        features_df1 = without_nan_function.drop(['cell_name', 'imaging_modality', "neurotransmitter_clone", 'morphology_clone'], axis=1)
        non_features_df = without_nan_function.loc[:, ['cell_name', 'imaging_modality', "neurotransmitter_clone", 'morphology_clone']]
    features = scaler.fit_transform(features_df1.to_numpy())
    features_df = pd.DataFrame(features, columns=features_df1.columns)
    finished_df = pd.concat([non_features_df, features_df], axis=1)
    return finished_df


if __name__ == '__main__':
    # set variables
    np.set_printoptions(suppress=True)

    # set data path
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')

    # load cells
    train_cells1 = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem'], load_repaired=True)
    train_cells2 = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['prediction_project'],
                                                 load_repaired=True)
    train_cells = pd.concat([train_cells1, train_cells2])
    train_cells = train_cells.drop_duplicates(keep='first', inplace=False, subset='cell_name')
    train_cells_no_function = train_cells.loc[(train_cells.function == 'nan'), :]
    train_cells = train_cells.loc[(train_cells.function != 'nan'), :]
    train_cells = train_cells.loc[(~train_cells.function.isna()), :]
    train_cells = train_cells.reset_index(drop=True)
    train_cells = train_cells.loc[train_cells['cell_name'].isin(['cell_576460752684182585', 'cell_576460752723528109', 'cell_576460752734566521']), :]

    print('\nFINISHED LOADING CELLS\n')
    # predict_cells
    calculate_metric2df(train_cells, 'test', path_to_data, force_new=False, train_or_predict='train')
    train_df = load_train_data_df(path_to_data, 'train_complete_with_neg_controls')
    # predict_df = load_predict_data_df(path_to_data, 'test_predict_df')
