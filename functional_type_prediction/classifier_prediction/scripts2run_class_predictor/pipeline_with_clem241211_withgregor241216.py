from holoviews.plotting.bokeh.styles import font_size

from hindbrain_structure_function.functional_type_prediction.classifier_prediction.class_predictor import *
from itertools import product
from sklearn.metrics import f1_score


def send_slack_message(RECEIVER="Florian Kämpf", MESSAGE="Script finished!"):
    slack_token = "xoxb-2212881652034-3363495253589-2kSTt6BcH3YTJtb3hIjsOJDp"
    client = WebClient(token=slack_token)
    ul = client.users_list()
    ul['real_name']
    member_list = []
if __name__ == "__main__":
    # load metrics and cells
    with_neurotransmitter = class_predictor(Path('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud'))
    with_neurotransmitter.load_cells_df(kmeans_classes=True, new_neurotransmitter=True,
                                        modalities=['pa', 'clem241211', 'em', 'clem_predict241211'], neg_control=True,
                                        input_em=True)
    with_neurotransmitter.calculate_metrics('FINAL_CLEM_CLEMPREDICT_EM_with_clem241211_withgregor250220')

    # with_neurotransmitter.calculate_published_metrics()
    with_neurotransmitter.load_cells_features('FINAL_CLEM_CLEMPREDICT_EM_with_clem241211_withgregor250220',
                                              with_neg_control=True,
                                              drop_neurotransmitter=False)
    # throw out truncated, exits and growth cone
    with_neurotransmitter.remove_incomplete()
    # apply gregors manual morphology annotations
    with_neurotransmitter.add_new_morphology_annotation()
    # select features
    # test.select_features_RFE('all', 'clem', cv=False,cv_method_RFE='lpo') #runs through all estimator
    with_neurotransmitter.select_features_RFE('all', 'clem', cv=False, save_features=True,
                                              estimator=AdaBoostClassifier(random_state=0), cv_method_RFE='ss',
                                              metric='f1')  # RidgeClassifier(random_state=0) Perceptron(random_state=0) AdaBoostClassifier(random state=0)|
    # select classifiers for the confusion matrices
    clf_fk = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    n_estimators_rf = 100
    clf_pv = RandomForestClassifier(n_estimators=n_estimators_rf)
    clf_ps = RandomForestClassifier(n_estimators=n_estimators_rf)
    clf_ff = RandomForestClassifier(n_estimators=n_estimators_rf)
    # make confusion matrices
    with_neurotransmitter.confusion_matrices(clf_fk, method='lpo', plot_cm_order_jon=True)
    # predict cells
    with_neurotransmitter.predict_cells(use_jon_priors=False,
                                        suffix='_optimize_all_predict')  # optimize_all_predict means to go for the 82.05%, alternative is balance_all_pa which goes to 79.49% ALL and 69.75% PA

    with_neurotransmitter.calculate_verification_metrics(calculate_smat=False, with_kunst=False,
                                                         required_tests=['IF', 'LOF'],
                                                         force_new=True)

    # optimal 'NBLAST_g','NBLAST_z','NBLAST_ak',
    # jon satisfied and gregors IIs 'NBLAST_g', 'NBLAST_z', 'MWU'
    # 'NBLAST_ak', 'NBLAST_g','LOF' gets cells for evertyhing but not gregors IIs
    # If IF has to be left out NBLAST_z, NBLAST_ak is best
    # 'MWU',"NBLAST_g",'LOF' would have all three and gives satstfying results deltaF1 is


    print("G II")
    print(with_neurotransmitter.prediction_predict_df.loc[
              with_neurotransmitter.prediction_predict_df['cell_name'].isin(['89189', '137722', '149747', '119243']), [
                  'cell_name', 'prediction', 'prediction_scaled', 'passed_tests']])
    print("Search for DT")
    print(with_neurotransmitter.prediction_predict_df.loc[
              with_neurotransmitter.prediction_predict_df['cell_name'].isin(['102596', '147009', '166876', '172045']), [
                  'cell_name', 'prediction', 'prediction_scaled', 'passed_tests']])

    print('G Cell types')
    print(with_neurotransmitter.prediction_predict_df.loc[
              with_neurotransmitter.prediction_predict_df.imaging_modality == 'EM'].groupby('prediction')[
              'passed_tests'].sum())
    print('J Cell types')
    print(with_neurotransmitter.prediction_predict_df.loc[
              with_neurotransmitter.prediction_predict_df.imaging_modality == 'clem'].groupby('prediction')[
              'passed_tests'].sum())


    with_neurotransmitter.plot_neurons('EM',
                                       output_filename='EM_predicted_optimize_all_predict.html')
    with_neurotransmitter.plot_neurons('EM',
                                       output_filename='EM_predicted_only_pass_tests.html',
                                       only_pass=True)

    with_neurotransmitter.plot_neurons('clem', output_filename='CLEM_predicted_optimize_all_predict.html')
    with_neurotransmitter.plot_neurons('clem', output_filename='CLEM_predicted_only_pass_tests.html', only_pass=True)

    # little code showing that on gregor cells ada confirms more with my manual classification

    eva = pd.read_excel(
        '/Users/fkampf/Documents/hindbrain_structure_function/nextcloud/manual_evaluation_all_cells_fk.xlsx')
    eva['correct_ada'] = np.where(eva['adaboost'] == eva['fk_cell_type'], eva['fk_rating_3.good_1.bad'], 0)
    eva['correct_perceptron'] = np.where(eva['perceptron'] == eva['fk_cell_type'], eva['fk_rating_3.good_1.bad'], 0)

    eva_tests_per = eva[eva['passed_tests_perceptron']]
    eva_tests_ada = eva[eva['passed_tests_ada']]

    eva_proc = eva.groupby(['imaging_modality', 'fk_cell_type'])[
        ['correct_ada', 'correct_perceptron']].sum().reset_index(drop=False)
    eva_tests_per_proc = eva_tests_per.groupby(['imaging_modality', 'fk_cell_type'])[
        ['correct_ada', 'correct_perceptron']].sum().reset_index(drop=False)
    eva_tests_ada_proc = eva_tests_ada.groupby(['imaging_modality', 'fk_cell_type'])[
        ['correct_ada', 'correct_perceptron']].sum().reset_index(drop=False)

    eva_fk_groundtruth = eva['fk_cell_type'].value_counts().reset_index()
    eva_proc['normed_correct_ada'] = eva_proc['correct_ada'] / eva_proc['fk_cell_type'].map(
        eva_fk_groundtruth.set_index('fk_cell_type')['count'])

    eva_proc['normed_correct_perceptron'] = eva_proc['correct_perceptron'] / eva_proc['fk_cell_type'].map(
        eva_fk_groundtruth.set_index('fk_cell_type')['count'])

    eva_tests_per_proc['normed_correct_ada'] = eva_tests_per_proc['correct_ada'] / eva_tests_per_proc[
        'fk_cell_type'].map(
        eva_fk_groundtruth.set_index('fk_cell_type')['count'])

    eva_tests_per_proc['normed_correct_perceptron'] = eva_tests_per_proc['correct_perceptron'] / eva_tests_per_proc[
        'fk_cell_type'].map(
        eva_fk_groundtruth.set_index('fk_cell_type')['count'])

    eva_tests_ada_proc['normed_correct_ada'] = eva_tests_ada_proc['correct_ada'] / eva_tests_ada_proc[
        'fk_cell_type'].map(
        eva_fk_groundtruth.set_index('fk_cell_type')['count'])

    eva_tests_ada_proc['normed_correct_perceptron'] = eva_tests_ada_proc['correct_perceptron'] / eva_tests_ada_proc[
        'fk_cell_type'].map(
        eva_fk_groundtruth.set_index('fk_cell_type')['count'])
