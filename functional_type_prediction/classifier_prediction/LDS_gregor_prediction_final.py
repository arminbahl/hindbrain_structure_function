from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction_with_neutral import *
import numpy as np
import navis
import pandas as pd
import matplotlib.pyplot as plt
import plotly
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_predict_jon_cells import *
from hindbrain_structure_function.functional_type_prediction.NBLAST.nblast_matrix_navis import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.make_dendrogramms import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction_with_neutral import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_predict_jon_cells import *
from hindbrain_structure_function.functional_type_prediction.NBLAST.nblast_matrix_navis import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.make_dendrogramms import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction_with_neutral import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from hindbrain_structure_function.functional_type_prediction.FK_tools.find_branches import *
import numpy as np
import pandas as pd
import plotly
# matplotlib.use('TkAgg')
from sklearn.svm import OneClassSVM
from scipy.stats import gaussian_kde
from scipy.stats import anderson_ksamp
import scipy.stats as stats
from colorama import Fore, Back, Style
import winsound

if __name__ == '__main__':
    #New segment: init constants
    color_dict = {
            "integrator ipsilateral": '#feb326b3',
            "integrator contralateral": '#e84d8ab3',
            "dynamic threshold": '#64c5ebb3',
            "motor command": '#7f58afb3',
        }
    acronym_dict = {'dynamic threshold': "DT", 'integrator contralateral': "CI", 'integrator ipsilateral': "II", 'motor command': "MC"}
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')
    brain_meshes = load_brs(path_to_data, 'raphe')
    solver = 'lsqr'
    shrinkage = 'auto'
    width_brain = 495.56

    #New segment: Prepare Train data
    use_k_means_classes=True
    use_new_neurotransmitter = True
    with_neg_control = True
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')

    #load cells, prepare and calculate metrics
    cells_train = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['pa','clem'], load_repaired=True)
    cells_train = prepare_data_4_metric_calc(cells_train,use_new_neurotransmitter,use_k_means_classes,path_to_data,train_or_predict='train')

    #cells_train.loc[:, 'swc'] = [navis.prune_twigs(x, '10 microns', recursive=True) for x in cells_train['swc']]
    calculate_metric2df(cells_train, 'FINAL', path_to_data, force_new=True, train_or_predict='train')

    #load preexisting metrics
    all_wn,column_labels_train_wn,train_df_wn = load_metrics_train('FINAL',path_to_data=path_to_data,with_neg_control=True)
    all, column_labels_train, train_df = load_metrics_train('FINAL', path_to_data=path_to_data, with_neg_control=False)

    #unpack metrics
    features_train, labels_train, labels_imaging_modality_train = all
    features_train_alt, labels_train_alt, labels_imaging_modality_train_alt, column_labels_train_alt, df_train_alt = load_train_data(path_to_data, file='FINAL_train')
    features_train_clem, labels_train_clem, labels_imaging_modality_train_clem = features_train[labels_imaging_modality_train == 'clem',:], labels_train[labels_imaging_modality_train == 'clem'],labels_imaging_modality_train[labels_imaging_modality_train == 'clem']
    features_train_wn, labels_train_wn, labels_imaging_modality_train_wn = all_wn
    print(Fore.RED + f"Train features like old: {(features_train_wn == features_train_alt).all()}")
    print(f"Predict DF like old: {train_df_wn.equals(df_train_alt)}\n")
    print(Fore.BLACK)

    features_neg_control =  features_train_wn[(train_df_wn.function == 'neg control').to_numpy(), :]
    labels_neg_control =  labels_train_wn[(train_df_wn.function == 'neg control').to_numpy()]

    #New segment: Prepare PREDICT DATA

    #load cells, prepare and calculate metrics
    cells_predict = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['em'], load_repaired=True)


    cells_predict = prepare_data_4_metric_calc(cells_predict,use_new_neurotransmitter,use_k_means_classes,path_to_data,train_or_predict='predict')
    for i,cell in cells_predict.iterrows():


        if (cell.swc.nodes.x.to_numpy()>=(width_brain/2)).any():
            cells_predict.loc[i,'morphology_n'] = 'contralateral'
        else:
            cells_predict.loc[i, 'morphology_n'] = 'ipsilateral'

    #cells_predict.loc[:,'swc'] = [navis.prune_twigs(x, '10 microns', recursive=True) for x in cells_predict['swc']]
    calculate_metric2df(cells_predict, 'FINAL', path_to_data, force_new=True, train_or_predict='predict')

    #load preexisting metrics
    all,column_labels,predict_df = load_metrics_predict('FINAL',path_to_data)


    #unpack metrics
    features_predict, labels_imaging_modality_predict = all
    features_predict_alt, labels_imaging_modality_predict_alt, column_labels_predict_alt, df_predict_alt, cell_names_predict_alt = load_predict_data(path_to_data, 'FINAL_predict')
    cells_predict = cells_predict.set_index('cell_name').loc[predict_df['cell_name']].reset_index()
    print(Fore.CYAN + f'Both DF are sorted the same: {(np.array([predict_df.cell_name,cells_predict.cell_name]).T[:,0] == np.array([predict_df.cell_name,cells_predict.cell_name]).T[:,1]).all()}')


    print(Fore.RED + f"Predict features like old: {(features_predict == features_predict_alt).all()}")
    print(f"Predict DF like old: {predict_df.equals(df_predict_alt)}")
    print(Fore.BLACK)




    #New segment: Select features

    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features_train, labels_train,
                                                                                             features_train_clem, labels_train_clem,
                                                                                             test_mod='CLEM', train_mod="ALL", plot=True, which_selection=DecisionTreeClassifier(), use_assessment_per_class=False,
                                                                                             use_std_scale=True)
    turned_on_features0 = ['morphology',
                          'x_extent', 'y_extent', 'z_extent',
                          'x_avg', 'y_avg', 'z_avg',
                          'soma_x', 'soma_y' 'soma_z',
                          'cable_length_2_first_branch', 'biggest_branch_longest_neurite',
                          'max_y_ipsi', 'min_y_ipsi']

    turned_on_features1 = ['morphology', 'neurotransmitter', 'x_extent', 'x_avg', 'y_avg',
       'soma_x', 'sholl_distance_max_branches_cable_length',
       'sholl_distance_max_branches_geosidic_cable_length',
       'main_path_total_branch_length']


    turned_on_features2 = ['morphology', 'neurotransmitter', 'contralateral_branches',
       'ipsilateral_branches', 'cable_length', 'bbox_volume', 'x_extent',
       'y_extent', 'z_extent', 'x_avg', 'y_avg', 'z_avg', 'soma_x',
       'soma_y', 'soma_z', 'n_leafs', 'n_branches', 'n_ends', 'n_edges',
       'main_branchpoint', 'n_persistence_points', 'max_strahler_index',
       'sholl_distance_max_branches',
       'sholl_distance_max_branches_cable_length',
       'sholl_distance_max_branches_geosidic',
       'sholl_distance_max_branches_geosidic_cable_length',
       'main_path_longest_neurite', 'main_path_total_branch_length',
       'first_major_branch_longest_neurite',
       'first_major_branch_total_branch_length',
       'first_branch_longest_neurite', 'first_branch_total_branch_length',
       'cable_length_2_first_branch', 'z_distance_first_2_first_branch',
       'biggest_branch_longest_neurite',
       'biggest_branch_total_branch_length', 'longest_connected_path',
       'n_nodes_ipsi_hemisphere', 'n_nodes_contra_hemisphere',
       'x_location_index', 'fraction_contra', 'angle', 'angle2d',
       'x_cross', 'y_cross', 'z_cross']

    turned_on_features3 = list(np.array(column_labels_train)[reduced_features_index]) + ['y_cross']

    turned_on_features = turned_on_features3
    # turned_on_features = turned_on_features3




    copy_index = [True if x in turned_on_features else False for x in np.array(column_labels)]
    # reduced_features_index = copy_index

    #New segment: Test that no ipsi get predicted as contra and vice versa
    priors = [len(labels_train[labels_train == x]) / len(labels_train) for x in np.unique(labels_train)]
    # priors = None
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors).fit(features_train, labels_train.flatten())
    clf_reduced = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors).fit(features_train[:,reduced_features_index], labels_train.flatten())

    predict_df['prediction'] = clf.predict(features_predict)
    cells_predict['prediction'] = clf.predict(features_predict)
    print(Fore.GREEN+'\n')

    print(predict_df.groupby(['morphology_clone', 'prediction']).size())
    print(Style.RESET_ALL+'\n')

    #New segment: NBlast

    cells_neg_control = cells_train.loc[(cells_train['function'] == 'neg_control') | (cells_train['function'] == 'neg control'), :]
    cells_train = cells_train.loc[(~(cells_train['function'] == 'neg_control')) & (~(cells_train['function'] == 'neg control')), :]

    #select cells via nblast
    names_dt = cells_train.loc[(cells_train['function']=='dynamic_threshold')|(cells_train['function']=='dynamic threshold'),'cell_name']
    names_ii = cells_train.loc[(cells_train['function']=='integrator')&(cells_train['morphology']=='ipsilateral'),'cell_name']
    names_ci = cells_train.loc[(cells_train['function']=='integrator')&(cells_train['morphology']=='contralateral'),'cell_name']
    names_mc = cells_train.loc[(cells_train['function']=='motor_command')|(cells_train['function']=='motor command'),'cell_name']
    names_nc = cells_neg_control['cell_name']

    smat_fish = load_zebrafish_nblast_matrix(return_smat_obj=True, prune=False, modalities=['clem', 'pa'],path_to_data=path_to_data)

    nb_train = nblast_two_groups_custom_matrix(cells_train,cells_train,custom_matrix=smat_fish,shift_neurons=False)
    nb_train_nc = nblast_two_groups_custom_matrix(cells_train, cells_neg_control, custom_matrix=smat_fish, shift_neurons=False)
    nb_train_predict = nblast_two_groups_custom_matrix(cells_train, cells_predict, custom_matrix=smat_fish, shift_neurons=False)

    nb_matches_cells_train = navis.nbl.extract_matches(nb_train, 2)
    nb_matches_cells_nc = navis.nbl.extract_matches(nb_train_nc.T, 2)
    nb_matches_cells_predict = navis.nbl.extract_matches(nb_train_predict.T, 2)

    nblast_values_dt = navis.nbl.extract_matches(nb_train.loc[names_dt,names_dt], 2)
    nblast_values_ii = navis.nbl.extract_matches(nb_train.loc[names_ii,names_ii], 2)
    nblast_values_ci = navis.nbl.extract_matches(nb_train.loc[names_ci,names_ci], 2)
    nblast_values_mc = navis.nbl.extract_matches(nb_train.loc[names_mc,names_mc], 2)

    z_score_dt = lambda x: abs((x-np.mean(list(nblast_values_dt.score_2)))/np.std(list(nblast_values_dt.score_2)))
    z_score_ii = lambda x: abs((x-np.mean(list(nblast_values_ii.score_2)))/np.std(list(nblast_values_ii.score_2)))
    z_score_ci = lambda x: abs((x-np.mean(list(nblast_values_ci.score_2)))/np.std(list(nblast_values_ci.score_2)))
    z_score_mc = lambda x: abs((x-np.mean(list(nblast_values_mc.score_2)))/np.std(list(nblast_values_mc.score_2)))

    cutoff = nb_matches_cells_train.loc[:,'score_2'].quantile(.1)
    print(f'{(nb_matches_cells_nc["score_1"] >= cutoff).sum()} of {nb_matches_cells_nc.shape[0]} neg_control cells pass NBlast general test.')

    subset_predict_cells = list(nb_matches_cells_predict.loc[nb_matches_cells_predict['score_1'] >= cutoff, 'id'])

    #New segment: Add data to predict df that is needed

    #order predict_df like cells_predict


    predict_df['metadata_path'] = cells_predict['metadata_path']
    predict_df['swc'] = cells_predict['swc']
    #New segment: Outlier detection

    OCSVM = OneClassSVM(gamma='scale', kernel='poly').fit(features_train)
    OCSVM_reduced =  OneClassSVM(gamma='scale',kernel='poly').fit(features_train[:,reduced_features_index])
    IF = IsolationForest(contamination=0.1, random_state=42).fit(features_train)
    IF_reduced = IsolationForest(contamination=0.1,random_state=42).fit(features_train[:,reduced_features_index])
    LOF = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(features_train)
    LOF_reduced = LocalOutlierFactor(n_neighbors=5,novelty=True).fit(features_train[:,reduced_features_index])

    predict_df.loc[:,'OCSVM'] = OCSVM.predict(features_predict) == 1
    predict_df.loc[:, 'OCSVM_reduced'] = OCSVM_reduced.predict(features_predict[:,reduced_features_index]) == 1
    predict_df.loc[:, 'IF'] = IF.predict(features_predict) == 1
    predict_df.loc[:, 'IF_reduced'] = IF_reduced.predict(features_predict[:,reduced_features_index]) == 1
    predict_df.loc[:, 'LOF'] = LOF.predict(features_predict) == 1
    predict_df.loc[:, 'LOF_reduced'] = LOF_reduced.predict(features_predict[:,reduced_features_index]) == 1


    print(f'{round(((OCSVM.predict(features_train) == 1).sum() / features_train.shape[0]) * 100,2)}% of train passes OCSVM')
    print(f'{round(((OCSVM.predict(features_neg_control) == 1).sum() / features_neg_control.shape[0]) * 100,2)}% of control passes OCSVM\n')

    print(f'{round(((OCSVM_reduced.predict(features_train[:,reduced_features_index]) == 1).sum() / features_train.shape[0]) * 100,2)}% of train passes OCSVM reduced')
    print(f'{round(((OCSVM_reduced.predict(features_neg_control[:,reduced_features_index]) == 1).sum() / features_neg_control.shape[0]) * 100,2)}% of control passes OCSVM reduced\n')

    print(f'{round(((IF.predict(features_train) == 1).sum() / features_train.shape[0]) * 100,2)}% of train passes IF')
    print(f'{round(((IF.predict(features_neg_control) == 1).sum() / features_neg_control.shape[0]) * 100,2)}% of control passes IF\n')

    print(f'{round(((IF_reduced.predict(features_train[:,reduced_features_index]) == 1).sum() / features_train.shape[0]) * 100,2)}% of train passes IF reduced')
    print(f'{round(((IF_reduced.predict(features_neg_control[:,reduced_features_index]) == 1).sum() / features_neg_control.shape[0]) * 100,2)}% of control passes IF reduced\n')

    print(f'{round(((LOF.predict(features_train) == 1).sum() / features_train.shape[0]) * 100,2)}% of train passes LOF')
    print(f'{round(((LOF.predict(features_neg_control) == 1).sum() / features_neg_control.shape[0]) * 100,2)}% of control passes LOF\n')

    print(f'{round(((LOF_reduced.predict(features_train[:,reduced_features_index]) == 1).sum() / features_train.shape[0]) * 100,2)}% of train passes LOF reduced')
    print(f'{round(((LOF_reduced.predict(features_neg_control[:,reduced_features_index]) == 1).sum() / features_neg_control.shape[0]) * 100,2)}% of control passes LOF reduced\n')



    # Combine your inliers and outliers data
    X_inliers = features_train[:,reduced_features_index]  # Shape: (120, 68)
    X_outliers = features_neg_control[:,reduced_features_index]  # Shape: (11, 68)

    # Create labels: 0 for inliers, 1 for outliers
    y_inliers = np.ones(X_inliers.shape[0])
    y_outliers = np.ones(X_outliers.shape[0])*-1

    # Combine the datasets
    X_combined = np.vstack((X_inliers, X_outliers))  # Shape: (131, 68)
    y_combined = np.hstack((y_inliers, y_outliers))  # Shape: (131,)

    # Split into training and testing data
    X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

    # Train a binary classifier (SVM)
    clf = OneClassSVM(kernel='rbf', gamma='auto')
    clf.fit(X_train_combined, y_train_combined)

    # Predict on the test set
    y_pred = clf.predict(X_test_combined)

    # Evaluate the classifier
    print(classification_report(y_test_combined, y_pred))

    #New segment: prediction


    #classifier
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors).fit(features_train, labels_train.flatten())
    clf_reduced = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors).fit(features_train[:,reduced_features_index], labels_train.flatten())

    color_list = []
    swc_list = []

    for i,cell in predict_df.iterrows():
        #prediction
        y_prob = clf.predict_proba(features_predict[i, :].reshape(1, -1))[0]
        y_pred = clf.predict(features_predict[i, :].reshape(1, -1))[0]

        y_prob_reduced = clf_reduced.predict_proba(features_predict[i, reduced_features_index].reshape(1, -1))[0]
        y_pred_reduced = clf_reduced.predict(features_predict[i,reduced_features_index].reshape(1, -1))[0]

        predict_df.loc[i, f'predicted'] = y_pred
        predict_df.loc[i, f'prob_{acronym_dict[clf.classes_[0]]}'] = y_prob[0] * 100
        predict_df.loc[i, f'prob_{acronym_dict[clf.classes_[1]]}'] = y_prob[1] * 100
        predict_df.loc[i, f'prob_{acronym_dict[clf.classes_[2]]}'] = y_prob[2] * 100
        predict_df.loc[i, f'prob_{acronym_dict[clf.classes_[3]]}'] = y_prob[3] * 100

        predict_df.loc[i, f'predicted_reduced'] = y_pred_reduced
        predict_df.loc[i, f'prob_reduced_{acronym_dict[clf.classes_[0]]}'] = y_prob_reduced[0] * 100
        predict_df.loc[i, f'prob_reduced_{acronym_dict[clf.classes_[1]]}'] = y_prob_reduced[1] * 100
        predict_df.loc[i, f'prob_reduced_{acronym_dict[clf.classes_[2]]}'] = y_prob_reduced[2] * 100
        predict_df.loc[i, f'prob_reduced_{acronym_dict[clf.classes_[3]]}'] = y_prob_reduced[3] * 100


        temp_title = (f'{acronym_dict[clf.classes_[0]]}: {"{:.5f}".format(y_prob[0] * 100)}%<br>'
                      f'{acronym_dict[clf.classes_[1]]}: {"{:.5f}".format(y_prob[1] * 100)}%<br>'
                      f'{acronym_dict[clf.classes_[2]]}: {"{:.5f}".format(y_prob[2] * 100)}%<br>'
                      f'{acronym_dict[clf.classes_[3]]}: {"{:.5f}".format(y_prob[3] * 100)}%<br>'
                      f'{acronym_dict[clf_reduced.classes_[0]]}_reduced: {"{:.5f}".format(y_prob[0] * 100)}%<br>'
                      f'{acronym_dict[clf_reduced.classes_[1]]}_reduced: {"{:.5f}".format(y_prob[1] * 100)}%<br>'
                      f'{acronym_dict[clf_reduced.classes_[2]]}_reduced: {"{:.5f}".format(y_prob[2] * 100)}%<br>'
                      f'{acronym_dict[clf_reduced.classes_[3]]}_reduced: {"{:.5f}".format(y_prob[3] * 100)}%'
                      )

        temp_prediction_string = (f'{acronym_dict[clf.classes_[0]]}:{"{:.5f}".format(y_prob[0])} '
                                  f'{acronym_dict[clf.classes_[1]]}:{"{:.5f}".format(y_prob[1])} '
                                  f'{acronym_dict[clf.classes_[2]]}:{"{:.5f}".format(y_prob[2])} '
                                  f'{acronym_dict[clf.classes_[3]]}:{"{:.5f}".format(y_prob[3])}')



        temp_prediction_string_reduced = (f'{acronym_dict[clf_reduced.classes_[0]]}:{"{:.5f}".format(y_prob_reduced[0])} '
                                          f'{acronym_dict[clf_reduced.classes_[1]]}:{"{:.5f}".format(y_prob_reduced[1])} '
                                          f'{acronym_dict[clf_reduced.classes_[2]]}:{"{:.5f}".format(y_prob_reduced[2])} '
                                          f'{acronym_dict[clf_reduced.classes_[3]]}:{"{:.5f}".format(y_prob_reduced[3])}')

        prediction_string = f'prediction = "{temp_prediction_string}"\n'
        prediction_string_reduced = f'prediction_reduced = "{temp_prediction_string_reduced}"\n'
        nblast_test_string = f'nblast_test_passed = {cell["cell_name"] in subset_predict_cells}\n'
        predict_df.loc[i, 'nblast_general'] = cell["cell_name"] in subset_predict_cells
        nblast_specific0_result = eval(f"z_score_{acronym_dict[y_pred].lower()}")(nb_matches_cells_predict.loc[nb_matches_cells_predict['id'] == cell["cell_name"], 'score_1'].iloc[0]) <= 1.96
        predict_df.loc[i,'nblast_specific0'] = nblast_specific0_result
        nblast_test_specific0 = f'nblast_test_specific0_passed = {nblast_specific0_result}\n'

        target = nb_train.loc[eval(f"names_{acronym_dict[y_pred].lower()}"),eval(f"names_{acronym_dict[y_pred].lower()}")].to_numpy().flatten()
        kde_target = gaussian_kde(target, bw_method=0.5)
        x_target = (np.linspace(np.min(target)-1,
                         np.max(target)+1, 1000))
        dist_target = kde_target(x_target)

        source = nb_train_predict.loc[:,cell["cell_name"]]
        kde_source = gaussian_kde(source, bw_method=0.5)
        x_source = (np.linspace(np.min(source)-1,
                         np.max(source)+1, 1000))
        dist_source = kde_source(x_source)

        nblast_test_specific1_result = anderson_ksamp([target, source]).pvalue>=0.05
        predict_df.loc[i, 'nblast_specific1'] = nblast_specific0_result
        nblast_test_specific1 = f'nblast_test_specific1_passed = {nblast_test_specific1_result}\n'
        nblast_test_specific2_result = stats.ks_2samp(source, target).pvalue>=0.05
        predict_df.loc[i, 'nblast_specific2'] = nblast_specific0_result
        nblast_test_specific2 = f'nblast_test_specific2_passed = {nblast_test_specific2_result}\n'

        probability_passed_string = f'probability_test_passed = {(y_prob > 0.7).any()}\n'
        predict_df.loc[i, 'probability_passed'] = (y_prob > 0.7).any()
        probability_passed_string_reduced = f'probability_test_passed_reduced = {(y_prob_reduced>0.7).any()}\n'
        predict_df.loc[i, 'probability_passed_reduced'] = (y_prob_reduced > 0.7).any()

        novelty_passed_OCSVM = f'novelty_passed_OCSVM = {cell["OCSVM"] == 1}\n'
        novelty_passed_OCSVM_reduced = f'novelty_passed_OCSVM_reduced = {cell["OCSVM_reduced"] == 1}\n'
        novelty_passed_IF = f'novelty_passed_IF = {cell["IF"] == 1}\n'
        novelty_passed_IF_reduced = f'novelty_passed_IF_reduced = {cell["IF_reduced"] == 1}\n'
        novelty_passed_LOF = f'novelty_passed_LOF = {cell["LOF"] == 1}\n'
        novelty_passed_LOF_reduced = f'novelty_passed_LOF_reduced = {cell["LOF_reduced"] == 1}\n'

        meta_paths = cell['metadata_path']
        if type(meta_paths) != list:
            meta_paths = [meta_paths]
        for meta_path in meta_paths:
            f = open(meta_path, 'r')
            t = f.read()
            if not t[-1:] == '\n':
                t = t + '\n'

            new_t = (t + prediction_string + prediction_string_reduced +
                     nblast_test_string + nblast_test_specific0 + nblast_test_specific1 + nblast_test_specific2 +
                     probability_passed_string + probability_passed_string_reduced +
                     novelty_passed_OCSVM + novelty_passed_OCSVM_reduced +
                     novelty_passed_IF + novelty_passed_IF_reduced +
                     novelty_passed_LOF + novelty_passed_LOF_reduced
                     )
            f.close()

            f = open(str(meta_path)[:-4] + "_with_prediction.txt", 'w')
            f.write(new_t)
            f.close()


    evaluation_df = predict_df.loc[:,['morphology', 'neurotransmitter', 'OCSVM', 'OCSVM_reduced',
       'IF', 'IF_reduced', 'LOF', 'LOF_reduced', 'predicted','predicted_reduced',
       'nblast_general', 'nblast_specific0', 'nblast_specific1',
       'nblast_specific2', 'probability_passed', 'probability_passed_reduced']]
    nblast_evaluate = evaluation_df.groupby(['predicted','nblast_general', 'nblast_specific0', 'nblast_specific1','nblast_specific2']).size().reset_index()
    reduced_methods_evaluate = evaluation_df.groupby(['predicted','IF_reduced', 'LOF_reduced', 'OCSVM_reduced']).size().reset_index()









    #New segment: plotting
    def restore_some(swc):

        temp_node_id = swc.nodes.loc[swc.nodes.type == 'root', 'node_id'].iloc[0]
        swc.soma = temp_node_id
        return swc


    predict_df['swc'] = predict_df['swc'].apply(lambda x: restore_some(x))

    cell_selection = predict_df.loc[(predict_df['nblast_general']),:]
    fig = navis.plot3d(list(cell_selection['swc']), backend='plotly', colors=[color_dict[x] for x in cell_selection['predicted']],
                       width=1920, height=1080, hover_name=True, alpha=1, title=temp_title)
    fig = navis.plot3d(brain_meshes, backend='plotly', fig=fig,
                       width=1920, height=1080, hover_name=True, title=temp_title)
    fig.update_layout(
        scene={
            'xaxis': {'autorange': 'reversed'},  # reverse !!!
            'yaxis': {'autorange': True},

            'zaxis': {'autorange': True},
            'aspectmode': "data",
            'aspectratio': {"x": 1, "y": 1, "z": 1}},
        title=dict(text='all_cells prediction', font=dict(size=20), automargin=True, yref='paper')
    )
    os.makedirs(path_to_data / 'make_figures_FK_output' / 'LDS_NBLAST_predictions_em', exist_ok=True)
    temp_file_name = path_to_data / 'make_figures_FK_output' / 'LDS_NBLAST_predictions_em' / f"all_cells_em_final.html"
    plotly.offline.plot(fig, filename=str(temp_file_name), auto_open=True, auto_play=False)

    os.makedirs(path_to_data / 'make_figures_FK_output' / 'LDS_NBLAST_predictions_em', exist_ok=True)

    #New segment: write excel

    predict_df.loc[:,['cell_name','neurotransmitter_clone', 'morphology_clone',
       'metadata_path', 'swc', 'OCSVM_reduced', 'IF_reduced',
       'LOF_reduced', 'predicted_reduced', 'prob_reduced_DT', 'prob_reduced_CI',
       'prob_reduced_II', 'prob_reduced_MC', 'nblast_general',
       'nblast_specific0', 'nblast_specific1', 'nblast_specific2',
       'probability_passed_reduced']].to_excel(path_to_data / 'make_figures_FK_output' / 'LDS_NBLAST_predictions_em'/'evaltuation_predict_df.xlsx')
    #New segment: print how many of manual predicted DTs are predicted as DT
    manual_predicted_DTs = [173141, 131678,133334,141963,146884,153284,168586,175440]
    manual_predicted_DTs = [str(x) for x in manual_predicted_DTs]
    print(f'{np.sum([x in list(cell_selection.loc[cell_selection["predicted_reduced"] == "dynamic threshold","cell_name"]) for x in manual_predicted_DTs])} of {len(manual_predicted_DTs)} manual predicted DTs predicted as DTs\n')
    print(str(cell_selection.loc[cell_selection['cell_name'].isin(manual_predicted_DTs)].groupby(['predicted_reduced']).size())+ "\n")
    print(cell_selection.loc[cell_selection['cell_name'].isin(manual_predicted_DTs),['prob_reduced_DT', 'prob_reduced_CI','prob_reduced_II', 'prob_reduced_MC']])
    # New segment: alarm
    duration = 200  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)
