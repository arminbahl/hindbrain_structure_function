import os

import matplotlib.pyplot as plt
import scipy
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.make_dendrogramms import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.find_branches import *
import winsound

from datetime import datetime
import plotly
import matplotlib
# matplotlib.use('TkAgg')
from tqdm import tqdm

if __name__ == "__main__":



    name_time = datetime.now()
    # Set the base path for data by reading from a configuration file; ensures correct data location is used.
    path_to_data = get_base_path()  # Ensure this path is set in path_configuration.txt

    all_cells = load_cells_predictor_pipeline(path_to_data=path_to_data,modalities=['pa',"clem"]) #

    # Define a dictionary mapping cell types to specific RGBA color codes for consistent visual representation.
    color_cell_type_dict = {
        "integrator_ipsi": (254, 179, 38, 0.7),
        "integrator_contra": (232, 77, 138, 0.7),
        "dynamic threshold": (100, 197, 235, 0.7),
        "motor command": (127, 88, 175, 0.7),
    }

    all_cells = all_cells[~all_cells['swc'].isna()]

    # fig = navis.plot3d([x for x in all_cells.iloc[:, -1]], backend='plotly',
    #                    width=3840, height=2160, hover_name=True, hover_id=False)
    # fig.update_layout(
    #     scene={
    #         'xaxis': {'autorange': 'reversed', 'zeroline': False, 'visible': False},  # reverse !!!
    #         'yaxis': {'autorange': True, 'zeroline': False, 'visible': False},
    #
    #         'zaxis': {'autorange': True, 'zeroline': False, 'visible': False},
    #         'aspectmode': "data",
    #         'aspectratio': {"x": 1, "y": 1, "z": 1}
    #     }
    # )
    # fig.update_layout(showlegend=False)
    # plotly.offline.plot(fig, filename="test.html", auto_open=True, auto_play=False)


    #split up cell_type_labels in own columns
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


    #compare within and between groups

    # within_ii,between_ii  = compute_nblast_within_and_between(all_cells,['ipsilateral','integrator'])
    # within_eii, between_eii = compute_nblast_within_and_between(all_cells, ['ipsilateral', 'integrator','excitatory'])
    #
    # within_ci,between_ci  = compute_nblast_within_and_between(all_cells,['contralateral','integrator'])
    # within_ici, between_ici = compute_nblast_within_and_between(all_cells, ['contralateral', 'integrator','inhibitory'])
    #
    # within_dt,between_dt  = compute_nblast_within_and_between(all_cells,['dynamic_threshold'])
    # within_mc, between_mc = compute_nblast_within_and_between(all_cells, ['motor_command'])
    #
    # all_whithin_between_values = [within_ii,between_ii,within_eii, between_eii,within_ci,between_ci,within_ici, between_ici,within_dt,between_dt,within_mc, between_mc]
    # all_whithin_between_values_str = ['within_ii', 'between_ii', 'within_eii', 'between_eii', 'within_ci', 'between_ci', 'within_ici', 'between_ici', 'within_dt', 'between_dt', 'within_mc', 'between_mc']
    #
    #
    # #plot barplots
    # plt.figure()
    # y_loc = np.max(np.concatenate(all_whithin_between_values)) * 1.1
    # for i,values in enumerate(all_whithin_between_values):
    #     plt.boxplot(values,positions=[i])
    #     if i %2 == 0:
    #         plt.plot([i, i + 1], [y_loc, y_loc], lw=3, c='k')
    #
    #
    #         plt.show()
    #         last = values
    #     elif i %2 == 1:
    #         t_stat, p_value = scipy.stats.ttest_ind(values,last)
    #         stat, p_value = scipy.stats.mannwhitneyu(values, last)
    #         sig = 'n.s.'
    #         if p_value < 0.05:
    #             sig = '*'
    #         if p_value < 0.01:
    #             sig = '**'
    #         if p_value < 0.001:
    #             sig = '***'
    #
    #         plt.text(i-0.5, y_loc, sig, ha='center', va='bottom',fontsize=15)
    #
    # plt.xticks(range(len(all_whithin_between_values_str)),labels=all_whithin_between_values_str,rotation=25)
    # plt.show()

    # make_dendrogramms(path_to_data,all_cells)
    if len(np.unique(all_cells.loc[:, "swc"].apply(lambda x: x.sampling_resolution)) )!= 1:

        all_cells.loc[:, "swc"] = all_cells.loc[:, "swc"].apply(lambda x: x.resample(0.1,inplace=False))

    #add cable length
    all_cells.loc[:, 'cable_length'] = all_cells.loc[:,"swc"].apply(lambda x: x.cable_length)
    #add bbox volume
    all_cells.loc[:, 'bbox_volume'] = all_cells.loc[:, "swc"].apply(lambda x: (x.extents[0]) * (x.extents[1]) * (x.extents[2]))
    # add x_extenct
    all_cells.loc[:, 'x_extent'] = all_cells.loc[:, "swc"].apply(lambda x: x.extents[0])
    # add y_extenct
    all_cells.loc[:, 'y_extent'] = all_cells.loc[:, "swc"].apply(lambda x: x.extents[1])
    # add z_extenct
    all_cells.loc[:, 'z_extent'] = all_cells.loc[:, "swc"].apply(lambda x: x.extents[2])

    #add avg x,y,z coordinate
    all_cells.loc[:,'x_avg'] = all_cells.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.x))
    all_cells.loc[:, 'y_avg'] = all_cells.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.y))
    all_cells.loc[:, 'z_avg'] = all_cells.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.z))

    #add soma x,y,z coordinate
    all_cells.loc[:,'soma_x'] = all_cells.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.loc[0,"x"]))
    all_cells.loc[:, 'soma_y'] = all_cells.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.loc[0,"y"]))
    all_cells.loc[:, 'soma_z'] = all_cells.loc[:, "swc"].apply(lambda x: np.mean(x.nodes.loc[0,"z"]))


    #add n_leafs
    all_cells.loc[:,'n_leafs'] = all_cells.loc[:, "swc"].apply(lambda x: x.n_leafs)
    # add n_branches
    all_cells.loc[:, 'n_branches'] = all_cells.loc[:, "swc"].apply(lambda x: x.n_branches)
    #add n_ends
    all_cells.loc[:,"n_ends"] = all_cells.loc[:, "swc"].apply(lambda x: x.n_ends)
    #add n_edges
    all_cells.loc[:,"n_edges"] = all_cells.loc[:, "swc"].apply(lambda x: x.n_edges)
    #main brainchpoint
    all_cells.loc[:, "main_branchpoint"] = all_cells.loc[:, "swc"].apply(lambda x: navis.find_main_branchpoint(x))

    # number of persitence points n_persistence_points
    all_cells.loc[:, "n_persistence_points"] =all_cells.loc[:, "swc"].apply(lambda x: len(navis.persistence_points(x)))
    #add strahler index
    _ =all_cells.loc[:, "swc"].apply(lambda x: navis.strahler_index(x))
    #add max strahler index
    all_cells.loc[:, "max_strahler_index"] = all_cells.loc[:, "swc"].apply(lambda x: x.nodes.strahler_index.max())

    #add sholl distance most bracnhes
    all_cells.loc[:,"sholl_distance_max_branches"] = all_cells.loc[:, "swc"].apply(lambda x: navis.sholl_analysis(x, radii=np.arange(10, 200, 5), center='root').branch_points.idxmax())

    # add sholl distance most bracnhes
    all_cells.loc[:, "sholl_distance_max_branches"] = all_cells.loc[:, "swc"].apply(lambda x: navis.sholl_analysis(x, radii=np.arange(10, 200, 10), center='root').branch_points.idxmax())
    all_cells.loc[:, "sholl_distance_max_branches_cable_length"] = all_cells.loc[:, ['sholl_distance_max_branches',"swc"]].apply(lambda x: navis.sholl_analysis(x['swc'],radii= np.arange(10,200,10),center='root', geodesic=False).cable_length[x['sholl_distance_max_branches']],axis=1)
    # add sholl distance most bracnhes
    all_cells.loc[:, "sholl_distance_max_branches_geosidic"] = all_cells.loc[:, "swc"].apply(lambda x: navis.sholl_analysis(x, radii=np.arange(10, 200, 10), center='root', geodesic=True).branch_points.idxmax())
    all_cells.loc[:, "sholl_distance_max_branches_geosidic_cable_length"] = all_cells.loc[:, ['sholl_distance_max_branches_geosidic',"swc"]].apply(lambda x: navis.sholl_analysis(x['swc'],radii= np.arange(10,200,10),center='root', geodesic=False).cable_length[x['sholl_distance_max_branches_geosidic']],axis=1)

    for i,cell in tqdm(all_cells.iterrows(),leave=False,total=len(all_cells)):
        temp = find_branches(cell['swc'].nodes,cell.cell_name)
        if not 'branches_df' in globals():
            branches_df = temp
        else:
            branches_df = pd.concat([branches_df,temp])

    width_brain = 495.56
    for i,cell in all_cells.iterrows():
        all_cells.loc[i,"main_path_longest_neurite"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name)&
                                                                          (branches_df['main_path'])&
                                                                          (branches_df['end_type']!='end'), 'longest_neurite_in_branch'].iloc[0]
        all_cells.loc[i,"main_path_total_branch_length"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name)&
                                                                              (branches_df['main_path'])&
                                                                              (branches_df['end_type']!='end'), 'total_branch_length'].iloc[0]

        try:
            all_cells.loc[i, "first_major_branch_longest_neurite"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name) &
                                                                                     (~branches_df['main_path']) &
                                                                                     (branches_df['end_type'] != 'end') &
                                                                                     (branches_df['total_branch_length'] >= 50), 'longest_neurite_in_branch'].iloc[0]
        except:
            all_cells.loc[i, "first_major_branch_longest_neurite"] = 0
        try:
            all_cells.loc[i, "first_major_branch_total_branch_length"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name) &
                                                                                         (~branches_df['main_path']) &
                                                                                         (branches_df['end_type'] != 'end') &
                                                                                         (branches_df['total_branch_length'] >= 50), 'total_branch_length'].iloc[0]
        except:
            all_cells.loc[i, "first_major_branch_total_branch_length"] = 0


        all_cells.loc[i,"first_branch_longest_neurite"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name)&
                                                                          (~branches_df['main_path'])&
                                                                          (branches_df['end_type']!='end'), 'longest_neurite_in_branch'].iloc[0]
        all_cells.loc[i,"first_branch_total_branch_length"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name)&
                                                                              (~branches_df['main_path'])&
                                                                              (branches_df['end_type']!='end'), 'total_branch_length'].iloc[0]
        #biggest major branch
        all_cells.loc[i, "biggest_branch_longest_neurite"] =  branches_df.loc[(branches_df['cell_name'] == cell.cell_name) &
                                                            (~branches_df['main_path']) &
                                                            (branches_df['end_type'] != 'end'), :].sort_values('total_branch_length', ascending=False)['longest_neurite_in_branch'].iloc[0]
        all_cells.loc[i,"biggest_branch_total_branch_length"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name)&
                                                                              (~branches_df['main_path'])&
                                                                              (branches_df['end_type']!='end'), 'total_branch_length'].iloc[0]

        all_cells.loc[i, "longest_connected_path"] = branches_df.loc[(branches_df['cell_name'] == cell.cell_name),'longest_connected_path'].iloc[0]

        all_cells.loc[i,'n_nodes_ipsi_hemisphere'] = (cell.swc.nodes.x<(width_brain/2)).sum()
        all_cells.loc[i, 'n_nodes_contra_hemisphere'] = (cell.swc.nodes.x < (width_brain / 2)).sum()


        def ic_index(x_coords):
            width_brain = 495.56

            distances = []
            for x in x_coords:
                distances.append(((width_brain/2) - x)/(width_brain/2))
            ipsi_contra_index = np.sum(distances)/len(distances)
            return ipsi_contra_index

        all_cells.loc[i, 'x_location_index'] = ic_index(cell.swc.nodes.x)

        all_cells.loc[i, 'fraction_contra'] = (cell.swc.nodes.x>(width_brain/2)).sum()/len(cell.swc.nodes.x)







            #neighboorhood component analysis
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    without_nan_function = all_cells.loc[(all_cells['function']!='nan'),:]

    features = np.array(without_nan_function.loc[:,without_nan_function.columns[26:]])
    without_nan_function.loc[:,'function'] = without_nan_function.loc[:, ['morphology','function']].apply(lambda x: x['function'].replace('_'," "), axis=1)
    labels = np.array(without_nan_function.loc[:, ['function']])
    # labels = np.array(without_nan_function.loc[:, ['morphology','function']] )
    # labels = labels[:,0] + labels[:,1]
    for i, item in enumerate(labels):
        if 'dynamic threshold' in item:
            labels[i] = 'dynamic threshold'
        if 'motor command' in item:
            labels[i] = 'motor command'

    X_train, X_test, y_train, y_test = train_test_split(features, labels,stratify=labels, test_size=0.7, random_state=42)

    nca = NeighborhoodComponentsAnalysis(random_state=42,n_components=2,init='lda')
    nca.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=5,weights='distance')
    knn.fit(X_train, y_train)


    print(knn.score(X_test, y_test))

    knn.fit(nca.transform(X_train), y_train)

    print(knn.score(nca.transform(X_test), y_test))


    #LMNN

    import numpy as np
    from metric_learn import LMNN
    from sklearn.datasets import load_iris

    features_normed = features/np.max(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=0.7, random_state=42)
    lmnn = LMNN(n_neighbors=3, learn_rate=1e-8,min_iter=1000,regularization=0.2,n_components=3)
    lmnn.fit(X_train, y_train)


    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_train, y_train)

    print(knn.score(X_test, y_test))
    knn.fit(lmnn.transform(X_train), y_train)
    print(knn.score(lmnn.transform(X_test), y_test))

    #LDS
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    #data
    features = np.array(without_nan_function.loc[:,without_nan_function.columns[26:]])
    without_nan_function.loc[:,'function'] = without_nan_function.loc[:, ['morphology','function']].apply(lambda x: x['function'].replace('_'," "), axis=1)
    labels = np.array(without_nan_function.loc[:, ['function']])
    # labels = np.array(without_nan_function.loc[:, ['morphology','function']] )
    # labels = labels[:,0] + labels[:,1]
    # for i, item in enumerate(labels):
    #     if 'dynamic threshold' in item:
    #         labels[i] = 'dynamic threshold'
    #     if 'motor command' in item:
    #         labels[i] = 'motor command'

    #create sets
    features_normed = features/np.max(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=0.3, random_state=42)

    clf = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')

    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)
    correct = prediction == y_test.flatten()
    percent_correct = correct.sum()/len(correct)
    print(percent_correct)



    #predict clem from only paGFP
    without_nan_function.loc[:, 'function'] = without_nan_function.loc[:, ['morphology', 'function']].apply(lambda x: x['function'].replace('_', " "), axis=1)
    features_paGFP = np.array(without_nan_function.loc[without_nan_function['imaging_modality']=='photoactivation', without_nan_function.columns[26:]])
    features_CLEM = np.array(without_nan_function.loc[without_nan_function['imaging_modality']=='clem', without_nan_function.columns[26:]])
    labels_paGFP = np.array(without_nan_function.loc[without_nan_function['imaging_modality']=='photoactivation', ['function']])
    labels_CLEM = np.array(without_nan_function.loc[without_nan_function['imaging_modality']=='clem', ['function']])

    # create sets
    features_normed = features / np.max(features)

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf = LinearDiscriminantAnalysis()

    clf.fit(features_paGFP, labels_paGFP)

    prediction = clf.predict(features_CLEM)
    correct = prediction == labels_CLEM.flatten()
    percent_correct = correct.sum() / len(correct)
    print("LDA: ",percent_correct)
    lmnn = LMNN(n_neighbors=3, learn_rate=1e-8, min_iter=1000, regularization=0.2, n_components=3)
    lmnn.fit(features_paGFP, labels_paGFP)

    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(features_paGFP, labels_paGFP)

    print("KNN Before: ",knn.score(features_CLEM, labels_CLEM))
    knn.fit(lmnn.transform(features_paGFP), labels_paGFP)
    print("LMNN: ",knn.score(lmnn.transform(features_CLEM), labels_CLEM))
    
    
    nca = NeighborhoodComponentsAnalysis(random_state=42,n_components=2,init='lda')
    nca.fit(features_paGFP, labels_paGFP)

    knn = KNeighborsClassifier(n_neighbors=5,weights='distance')
    knn.fit(features_paGFP, labels_paGFP)




    knn.fit(nca.transform(features_paGFP), labels_paGFP)

    print("NCA: ",knn.score(nca.transform(features_CLEM), labels_CLEM))

    





