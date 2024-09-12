from hindbrain_structure_function.functional_type_prediction.FK_tools.fragment_neurite import *

from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.find_branches import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly
# matplotlib.use('TkAgg')
from tqdm import tqdm


def calculate_metric(cell_df,file_name,path_to_data,force_new=False,train_or_predict='train',):
    print('\nSTARTED calculate_metric\n')
    width_brain = 495.56
    skip = False
    if Path(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5').exists():
            if 'predictor_pipeline_features' in h5py.File(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5').keys():
                if h5py.File(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5')['angle_cross/axis1'].shape[0] == cell_df.shape[0] and not force_new:
                    skip = True
    if not skip:
        #prune
        # cell_df.loc[:,'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in cell_df['swc']]
        # cell_df.loc[:,'swc'] = [navis.prune_twigs(x, 10, recursive=True) for x in cell_df['swc']]

        #reset index
        cell_df = cell_df.reset_index(drop=True)


        #predictor pipeline
        cell_df.loc[:, "contralateral_branches"] = cell_df.loc[:, "swc"].apply(lambda x: len(x.nodes.loc[(x.nodes.x > width_brain / 2) & (x.nodes.type == 'branch'), 'type']))
        cell_df.loc[:, "ipsilateral_branches"] = cell_df.loc[:, "swc"].apply(lambda x: len(x.nodes.loc[(x.nodes.x < width_brain / 2) & (x.nodes.type == 'branch'), 'type']))


        #add cable length
        cell_df.loc[:, 'cable_length'] = cell_df.loc[:,"swc"].apply(lambda x: x.cable_length)
        #add bbox volume
        cell_df.loc[:, 'bbox_volume'] = cell_df.loc[:, "swc"].apply(lambda x: (x.extents[0]) * (x.extents[1]) * (x.extents[2]))
        # add x_extenct
        cell_df.loc[:, 'x_extent'] = cell_df.loc[:, "swc"].apply(lambda x: x.extents[0])
        # add y_extenct
        cell_df.loc[:, 'y_extent'] = cell_df.loc[:, "swc"].apply(lambda x: x.extents[1])
        # add z_extenct
        cell_df.loc[:, 'z_extent'] = cell_df.loc[:, "swc"].apply(lambda x: x.extents[2])

        #add avg x,y,z coordinate
        cell_df.loc[:,'x_avg'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.x))
        cell_df.loc[:, 'y_avg'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.y))
        cell_df.loc[:, 'z_avg'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.z))

        #add soma x,y,z coordinate
        cell_df.loc[:,'soma_x'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.loc[0,"x"]))
        cell_df.loc[:, 'soma_y'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.loc[0,"y"]))
        cell_df.loc[:, 'soma_z'] = cell_df.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.loc[0,"z"]))


        #add n_leafs
        cell_df.loc[:,'n_leafs'] = cell_df.loc[:, "swc"].apply(lambda x: x.n_leafs)
        # add n_branches
        cell_df.loc[:, 'n_branches'] = cell_df.loc[:, "swc"].apply(lambda x: x.n_branches)
        #add n_ends
        cell_df.loc[:,"n_ends"] = cell_df.loc[:, "swc"].apply(lambda x: x.n_ends)
        #add n_edges
        cell_df.loc[:,"n_edges"] = cell_df.loc[:, "swc"].apply(lambda x: x.n_edges)
        #main brainchpoint
        cell_df.loc[:, "main_branchpoint"] = cell_df.loc[:, "swc"].apply(lambda x: navis.find_main_branchpoint(x))

        # number of persitence points n_persistence_points
        cell_df.loc[:, "n_persistence_points"] =cell_df.loc[:, "swc"].apply(lambda x: len(navis.persistence_points(x)))
        #add strahler index
        _ =cell_df.loc[:, "swc"].apply(lambda x: navis.strahler_index(x))
        #add max strahler index
        cell_df.loc[:, "max_strahler_index"] = cell_df.loc[:, "swc"].apply(lambda x: x.nodes.strahler_index.max())

        #add sholl distance most bracnhes
        cell_df.loc[:,"sholl_distance_max_branches"] = cell_df.loc[:, "swc"].apply(lambda x: navis.sholl_analysis(x, radii=np.arange(10, 200, 5), center='root').branch_points.idxmax())

        # add sholl distance most bracnhes
        cell_df.loc[:, "sholl_distance_max_branches"] = cell_df.loc[:, "swc"].apply(lambda x: navis.sholl_analysis(x, radii=np.arange(10, 200, 10), center='root').branch_points.idxmax())
        cell_df.loc[:, "sholl_distance_max_branches_cable_length"] = cell_df.loc[:, ['sholl_distance_max_branches',"swc"]].apply(lambda x: navis.sholl_analysis(x['swc'],radii= np.arange(10,200,10),center='root', geodesic=False).cable_length[x['sholl_distance_max_branches']],axis=1)
        # add sholl distance most bracnhes
        cell_df.loc[:, "sholl_distance_max_branches_geosidic"] = cell_df.loc[:, "swc"].apply(lambda x: navis.sholl_analysis(x, radii=np.arange(10, 200, 10), center='root', geodesic=True).branch_points.idxmax())
        cell_df.loc[:, "sholl_distance_max_branches_geosidic_cable_length"] = cell_df.loc[:, ['sholl_distance_max_branches_geosidic',"swc"]].apply(lambda x: navis.sholl_analysis(x['swc'],radii= np.arange(10,200,10),center='root', geodesic=False).cable_length[x['sholl_distance_max_branches_geosidic']],axis=1)
        branches_df = None
        for i,cell in tqdm(cell_df.iterrows(),leave=False,total=len(cell_df)):
            temp = find_branches(cell['swc'].nodes,cell.cell_name)
            if type(branches_df)== type(None):
                branches_df = temp
            else:
                branches_df = pd.concat([branches_df,temp])


        for i,cell in cell_df.iterrows():
            cell_df.loc[i,"main_path_longest_neurite"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name)&
                                                                              (branches_df['main_path'])&
                                                                              (branches_df['end_type']!='end'), 'longest_neurite_in_branch'].iloc[0]
            cell_df.loc[i,"main_path_total_branch_length"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name)&
                                                                                  (branches_df['main_path'])&
                                                                                  (branches_df['end_type']!='end'), 'total_branch_length'].iloc[0]

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

            fragmented_neuron = navis.split_into_fragments(cell_df.loc[i,"swc"],cell_df.loc[i,"swc"].n_leafs)
            cell_df.loc[i,"first_branch_longest_neurite"] = navis.longest_neurite(fragmented_neuron[1]).cable_length
            cell_df.loc[i,"first_branch_total_branch_length"] = fragmented_neuron[1].cable_length

            temp = cell_df.loc[i,"swc"]
            temp = navis.prune_twigs(temp, 5, recursive=True)
            temp_node_id = temp.nodes.loc[temp.nodes.type == 'branch', 'node_id'].iloc[0]
            temp = navis.cut_skeleton(temp, temp_node_id)
            cell_df.loc[i,"cable_length_2_first_branch"] =temp[1].cable_length
            cell_df.loc[i, "z_distance_first_2_first_branch"] = temp[1].nodes.iloc[0].z-temp[1].nodes.iloc[-1].z



            #biggest major branch
            cell_df.loc[i, "biggest_branch_longest_neurite"] =  branches_df.loc[(branches_df['cell_name'] == cell.cell_name) &
                                                                (~branches_df['main_path']) &
                                                                (branches_df['end_type'] != 'end'), :].sort_values('total_branch_length', ascending=False)['longest_neurite_in_branch'].iloc[0]
            cell_df.loc[i,"biggest_branch_total_branch_length"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name)&
                                                                                  (~branches_df['main_path'])&
                                                                                  (branches_df['end_type']!='end'), 'total_branch_length'].iloc[0]

            cell_df.loc[i, "longest_connected_path"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name),'longest_connected_path'].iloc[0]

            cell_df.loc[i,'n_nodes_ipsi_hemisphere'] = (cell.swc.nodes.x<(width_brain/2)).sum()
            cell_df.loc[i, 'n_nodes_contra_hemisphere'] = (cell.swc.nodes.x < (width_brain / 2)).sum()


            def ic_index(x_coords):
                width_brain = 495.56

                distances = []
                for x in x_coords:
                    distances.append(((width_brain/2) - x)/(width_brain/2))
                ipsi_contra_index = np.sum(distances)/len(distances)
                return ipsi_contra_index

            cell_df.loc[i, 'x_location_index'] = ic_index(cell.swc.nodes.x)

            cell_df.loc[i, 'fraction_contra'] = (cell.swc.nodes.x>(width_brain/2)).sum()/len(cell.swc.nodes.x)






        temp_index = list(cell_df.columns).index('contralateral_branches')
        temp = cell_df.loc[:, cell_df.columns[temp_index:]]

        temp.to_hdf(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5', 'predictor_pipeline_features')

        if train_or_predict == 'predict':
            temp = cell_df.loc[:,['cell_name','imaging_modality','morphology','neurotransmitter']]
        elif train_or_predict == 'train':
            temp = cell_df.loc[:, ['cell_name', 'imaging_modality', 'function', 'morphology', 'neurotransmitter']]
        temp.to_hdf(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5', 'function_morphology_neurotransmitter')
        print('\nFINISHED calculate_metric PART 1\n')
        print('\nSTART calculate_metric PART 2\n')
    skip = False
    if Path(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5').exists():
            if 'angle_cross' in h5py.File(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5').keys():
                if h5py.File(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5')['angle_cross/axis1'].shape[0] == cell_df.shape[0] and not force_new:
                    skip = True
    if not skip:
        #extract branching angle and coords of crossing for contralateral neurons
        for i,cell in tqdm(cell_df.iterrows(),total=cell_df.shape[0],leave=False):
            if cell.morphology == 'contralateral':
                #pass

                angle,crossing_coords,fragments_list = direct_angle_and_crossing_extraction(cell['swc'].nodes,projection="3d")
                angle2d, crossing_coords, fragments_list = direct_angle_and_crossing_extraction(cell['swc'].nodes, projection="2d")

                if np.isnan(angle):
                    pass
                try:
                    cell_df.loc[i,'angle'] = angle
                    cell_df.loc[i, 'angle2d'] = angle2d
                    cell_df.loc[i,'x_cross'] = crossing_coords[0]
                    cell_df.loc[i,'y_cross'] = crossing_coords[1]
                    cell_df.loc[i,'z_cross'] = crossing_coords[2]

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

        temp = cell_df.loc[:, ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross']]
        temp.to_hdf(path_to_data / 'make_figures_FK_output' / f'{file_name}_features.hdf5', 'angle_cross')
    print('\nFINISHED calculate_metric PART 2\n')
def load_train_data(path,file="CLEM_and_PA"):

    #file_path = path / 'make_figures_FK_output' / 'prediction_project_features.hdf5'

    file_path = path / 'make_figures_FK_output' / f'{file}_features.hdf5'
    fmn = pd.read_hdf(file_path, 'function_morphology_neurotransmitter')
    pp = pd.read_hdf(file_path, 'predictor_pipeline_features')
    ac = pd.read_hdf(file_path, 'angle_cross')

    all_cells = pd.concat([fmn, pp, ac], axis=1)

    # throw out weird jon cells
    # all_cells = all_cells.loc[~all_cells.cell_name.isin(["cell_576460752734566521", "cell_576460752723528109", "cell_576460752684182585"]), :]

    # Data Preprocessing
    without_nan_function = all_cells[all_cells['function'] != 'nan']
    without_nan_function = without_nan_function.loc[~without_nan_function['function'].isin(['off-response', 'no response', 'noisy, little modulation']), :]
    without_nan_function = without_nan_function.sort_values(by=['function', 'morphology', 'imaging_modality', 'neurotransmitter'])
    without_nan_function = without_nan_function.reset_index(drop=True)
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
    neurotransmitter2int_dict = {'excitatory': 0,'inhibitory': 1,'nan': 2,'na': 2}
    morphology2int_dict = {'contralateral': 0, 'ipsilateral': 1}

    for work_column in columns_replace_string:
        without_nan_function.loc[:,work_column+"_clone"] = without_nan_function[work_column]
        for key in eval(f'{work_column}2int_dict').keys():
            without_nan_function.loc[without_nan_function[work_column] == key, work_column] = eval(f'{work_column}2int_dict')[key]



    # Extract labels
    labels = without_nan_function['function'].to_numpy()
    labels_imaging_modality = without_nan_function['imaging_modality'].to_numpy()
    column_labels = list(without_nan_function.columns[3:-len(columns_replace_string)])
    # column_labels = list(without_nan_function.columns[~without_nan_function.columns.isin(['cell_name',"imaging_modality",'neurotransmitter','function'])])

    # Extract features
    features = without_nan_function.iloc[:, 3:-len(columns_replace_string)].to_numpy()
    # features = without_nan_function.loc[:,~without_nan_function.columns.isin(['cell_name',"imaging_modality",'neurotransmitter','function'])].to_numpy()

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, labels, labels_imaging_modality, column_labels, without_nan_function

def load_predict_data(path,file):
    file_path = path / 'make_figures_FK_output' / f'{file}_features.hdf5'
    fmn = pd.read_hdf(file_path, 'function_morphology_neurotransmitter')
    pp = pd.read_hdf(file_path, 'predictor_pipeline_features')
    ac = pd.read_hdf(file_path, 'angle_cross')

    all_cells = pd.concat([fmn, pp, ac], axis=1)

    # throw out weird jon cells
    all_cells = all_cells.loc[~all_cells.cell_name.isin(["cell_576460752734566521", "cell_576460752723528109", "cell_576460752684182585"]), :]
    cell_names = all_cells.cell_name
    # Data Preprocessing
    without_nan_function = all_cells
    without_nan_function = without_nan_function.sort_values(by=['morphology', 'imaging_modality', 'neurotransmitter'])
    without_nan_function = without_nan_function.reset_index(drop=True)

    # Impute NaNs
    columns_possible_nans = ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross']
    without_nan_function.loc[:, columns_possible_nans] = without_nan_function[columns_possible_nans].fillna(0)

    # Replace strings with indices
    columns_replace_string = ['neurotransmitter', 'morphology']
    neurotransmitter2int_dict = {'excitatory': 0,'inhibitory': 1,'nan': 2,'na': 2}
    morphology2int_dict = {'contralateral': 0, 'ipsilateral': 1}

    for work_column in columns_replace_string:
        without_nan_function.loc[:,work_column+"_clone"] = without_nan_function[work_column]
        for key in eval(f'{work_column}2int_dict').keys():
            without_nan_function.loc[without_nan_function[work_column] == key, work_column] = eval(f'{work_column}2int_dict')[key]

    # sort by function an imaging modality

    cell_names = without_nan_function.cell_name

    # Extract labels
    labels_imaging_modality = without_nan_function['imaging_modality'].to_numpy()
    column_labels = list(without_nan_function.columns[2:-len(columns_replace_string)])
    # Extract features
    features = without_nan_function.iloc[:, 2:-len(columns_replace_string)].to_numpy()
    # features = without_nan_function.loc[:,~without_nan_function.columns.isin(['cell_name',"imaging_modality",'neurotransmitter','function'])].to_numpy()

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, labels_imaging_modality, column_labels, without_nan_function,cell_names

if __name__ == '__main__':
    #set variables
    np.set_printoptions(suppress=True)

    #set data path
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')


    #load cells
    clem_predict_cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'),modalities=['clem_predict'],load_repaired=True)
    clem_pa_cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem','pa'], load_repaired=True)
    prediction_project_cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['prediction_project'], load_repaired=True)
    print('\nFINISHED LOADING CELLS\n')
    #predict_cells
    calculate_metric(clem_predict_cells,'clem_predict',path_to_data=path_to_data,force_new=True,train_or_predict='predict')
    calculate_metric(clem_pa_cells, 'CLEM_and_PA',path_to_data=path_to_data, force_new=True,train_or_predict='train')
    calculate_metric(prediction_project_cells,'prediction_project',path_to_data=path_to_data,force_new=True,train_or_predict='train')
    print('\nFINISHED CALCULATING METRICS\n')


    #Load data to train model

    features_train, labels_train, labels_imaging_modality_train, column_labels_train, df_train = load_train_data(path_to_data,file="CLEM_and_PA")


    #find reduced features
    reduced_features_train, reduced_features_index_train = determine_important_features_RFECV(features_train, labels_train, column_labels_train, scoring='roc_auc_ovo')
    reduced_features, reduced_features_index, collection_coef_matrix = determine_important_features(features_train, labels_train, column_labels_train, return_collection_coef_matrix=True)

    # features_train = features_train[:,reduced_features_index]

    #load predict data
    features_predict, labels_imaging_modality_predict, column_labels_predict, df_predict,cell_names = load_predict_data(path_to_data,'clem_predict')
    # features_predict = features_predict[:,['soma' not in x for x in column_labels_predict]]
    # column_labels_predict = np.array(column_labels_predict)[['soma' not in x for x in column_labels_predict]]
    # features_predict = features_predict[:, ['_avg' not in x for x in column_labels_predict]]
    # features_predict = features_predict[:,reduced_features_index]


    brain_meshes = load_brs(path_to_data, 'raphe')
    solver = 'lsqr'
    shrinkage = 'auto'
    priors = [len(labels_train[labels_train == x]) / len(labels_train) for x in np.unique(labels_train)]
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors)
    clf.fit(features_train[:,:], labels_train.flatten())
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    find_novelty = OneClassSVM( gamma='auto').fit(features_train)
    lof = LocalOutlierFactor(novelty=True).fit(features_train)


    # plt.figure(figsize=(15, 15))
    # plt.subplots_adjust(left=0.3, right=0.8, top=0.5, bottom=0.3)
    # aaa = plt.pcolormesh(clf.coef_[::-1])
    # plt.xticks(np.arange(0.5, clf.coef_.shape[1]), column_labels_train, rotation=60, fontsize='x-small',ha='right')
    # plt.yticks(np.arange(0.5, clf.coef_.shape[0]), clf.classes_[::-1])
    # plt.colorbar(aaa)
    # plt.show()



    how_many_identified = 0
    for i,cell_name in zip(range(features_predict.shape[0]),cell_names):
        y_pred = clf.predict(features_predict[i,:].reshape(1,-1))
        y_prob = clf.predict_proba(features_predict[i,:].reshape(1,-1))[0]
        nov = find_novelty.predict(features_predict[i,:].reshape(1,-1))[0]
        nov2 = lof.predict(features_predict[i,:].reshape(1,-1))[0]
        if nov == 1 and nov2==1:
            print(cell_name,'\n',clf.classes_[0],np.round(y_prob[0],4),clf.classes_[1],np.round(y_prob[1],4),clf.classes_[2],np.round(y_prob[2],4),clf.classes_[3],np.round(y_prob[3],4),'\n')
            temp_title = f'{clf.classes_[0]}:{np.round(y_prob[0],4)}\n{clf.classes_[1]}: {np.round(y_prob[1], 4)}\n{clf.classes_[2]}:{np.round(y_prob[2],4)}\n{clf.classes_[3]}:{np.round(y_prob[3], 4)}'
            if (y_prob>0.9).any():
                fig = navis.plot3d(clem_predict_cells.iloc[i].loc['swc'], backend='plotly',
                                   width=1920, height=1080, hover_name=True, alpha=1,title=temp_title)
                fig = navis.plot3d(brain_meshes, backend='plotly', fig=fig,
                                   width=1920, height=1080, hover_name=True,title=temp_title)
                fig.update_layout(
                    scene={
                        'xaxis': {'autorange': 'reversed'},  # reverse !!!
                        'yaxis': {'autorange': True},

                        'zaxis': {'autorange': True},
                        'aspectmode': "data",
                        'aspectratio': {"x": 1, "y": 1, "z": 1}},
                    title = dict(text=temp_title, font=dict(size=20), automargin=True, yref='paper')
                )
                plotly.offline.plot(fig, filename=f"{cell_name}.html", auto_open=True, auto_play=False)
                how_many_identified+=1

    #try svm

    rs = 42
    collection_prediction_correct = []
    collection_prediction_correct_SVM = []

    collected_coef = None
    collected_coef_SVM = None

    from sklearn.svm import SVC

    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(features_train, labels_train, stratify=labels_train, test_size=0.3, random_state=rs)
        rs+=1


        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors)
        clf.fit(X_train, y_train.flatten())
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        collection_prediction_correct.append(accuracy)

        if collected_coef is None:
            collected_coef = clf.coef_[np.newaxis, :, :]
        else:
            collected_coef = np.vstack([collected_coef, clf.coef_[np.newaxis, :, :]])

        clf = SVC(kernel='linear')
        clf.fit(X_train, y_train.flatten())
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        collection_prediction_correct_SVM.append(accuracy)

        if collected_coef_SVM is None:
            collected_coef_SVM = clf.coef_[np.newaxis, :, :]
        else:
            collected_coef_SVM = np.vstack([collected_coef_SVM, clf.coef_[np.newaxis, :, :]])

        # predict

    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(left=0.3, right=0.8, top=0.5, bottom=0.3)
    aaa = plt.pcolormesh(clf.coef_[::-1])
    plt.xticks(np.arange(0.5, clf.coef_.shape[1]), column_labels_train, rotation=60, fontsize='x-small',ha='right')
    #plt.yticks(np.arange(0.5, clf.coef_.shape[0]), clf.classes_[::-1])
    plt.colorbar(aaa)
    plt.show()




    #novelty detectuoibn
    from sklearn.ensemble import IsolationForest
    rs=42
    dict_test = {'iso':[],'fno':[],'lofo':[]}
    dict_predict = {'iso': [], 'fno': [], 'lofo': []}
    for i in tqdm(range(1000)):
        X_train, X_test, y_train, y_test = train_test_split(features_train, labels_train, stratify=labels_train, test_size=0.3, random_state=rs)
        rs+=1
        isolation_forest = IsolationForest(contamination='auto', random_state=42).fit(X_train)
        find_novelty = OneClassSVM(gamma='auto').fit(X_train)
        lof = LocalOutlierFactor(novelty=True).fit(X_train)

        iso = isolation_forest.predict(X_test)
        lofo = lof.predict(X_test)
        fno = find_novelty.predict(X_test)
        dict_test['iso'].append((iso == 1).sum() / X_train.shape[0])
        dict_test['fno'].append((fno == 1).sum() / X_train.shape[0])
        dict_test['lofo'].append((lofo == 1).sum() / X_train.shape[0])

        iso = isolation_forest.predict(features_predict)
        lofo = lof.predict(features_predict)
        fno = find_novelty.predict(features_predict)
        dict_predict['iso'].append((iso == 1).sum() / X_train.shape[0])
        dict_predict['fno'].append((fno == 1).sum() / X_train.shape[0])
        dict_predict['lofo'].append((lofo == 1).sum() / X_train.shape[0])

