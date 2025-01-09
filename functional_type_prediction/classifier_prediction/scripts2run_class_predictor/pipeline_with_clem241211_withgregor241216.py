from holoviews.plotting.bokeh.styles import font_size

from hindbrain_structure_function.functional_type_prediction.classifier_prediction.class_predictor import *
from itertools import product
from sklearn.metrics import f1_score

if __name__ == "__main__":
    # load metrics and cells
    with_neurotransmitter = class_predictor(Path('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud'))
    with_neurotransmitter.load_cells_df(kmeans_classes=True, new_neurotransmitter=True,
                                        modalities=['pa', 'clem241211', 'em', 'clem_predict241211'], neg_control=True,
                                        input_em=True)
    with_neurotransmitter.calculate_metrics('FINAL_CLEM_CLEMPREDICT_EM_with_clem241211_withgregor241216')

    # with_neurotransmitter.calculate_published_metrics()
    with_neurotransmitter.load_cells_features('FINAL_CLEM_CLEMPREDICT_EM_with_clem241211_withgregor241216',
                                              with_neg_control=True,
                                              drop_neurotransmitter=False)
    # throw out truncated, exits and growth cone
    with_neurotransmitter.remove_incomplete()
    # apply gregors manual morphology annotations
    with_neurotransmitter.add_new_morphology_annotation()
    # select features
    # test.select_features_RFE('all', 'clem', cv=False,cv_method_RFE='lpo') #runs through all estimator
    with_neurotransmitter.select_features_RFE('all', 'clem', cv=False, save_features=True,
                                              estimator=Perceptron(random_state=0), cv_method_RFE='ss',
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
                                                         required_tests=['NBLAST_g', 'NBLAST_z', 'CVM'], force_new=True)

    # optimal 'NBLAST_g','NBLAST_z','NBLAST_ak','NBLAST_ks'
    # jon satisfied and gregors IIs 'NBLAST_g', 'NBLAST_z', 'CVM'

    # 'NBLAST_ks_2samp_passed', 'IF','NBLAST_general_pass'
    # 'NBLAST_general_pass','NBLAST_zscore_pass','NBLAST_anderson_ksamp_passed','NBLAST_ks_2samp_passed'

    # this codes looks how much different the new metrics are
    # print('new_metrics')
    # print(with_neurotransmitter.prediction_predict_df[with_neurotransmitter.prediction_predict_df.function=='to_predict'].groupby('imaging_modality').size())
    # print(with_neurotransmitter.prediction_predict_df[with_neurotransmitter.prediction_predict_df.function=='to_predict'].groupby('imaging_modality')['passed_tests'].sum())

    # with_neurotransmitter.calculate_verification_metrics(calculate_smat=False, with_kunst=False,
    #                                                      required_tests=['NBLAST_ks_2samp_passed', 'IF','NBLAST_general_pass'], force_new=True)
    # print('old_metrics')
    # print(with_neurotransmitter.prediction_predict_df[with_neurotransmitter.prediction_predict_df.function=='to_predict'].groupby('imaging_modality').size())
    # print(with_neurotransmitter.prediction_predict_df[with_neurotransmitter.prediction_predict_df.function=='to_predict'].groupby('imaging_modality')['passed_tests'].sum())
    #
    #
    # print(with_neurotransmitter.prediction_predict_df.loc[
    #           with_neurotransmitter.prediction_predict_df['cell_name'].isin(['147009', '102596']), [
    #               'cell_name', 'prediction', 'prediction_scaled']])

    with_neurotransmitter.plot_neurons('EM',
                                       output_filename='EM_predicted_optimize_all_predict.html')
    with_neurotransmitter.plot_neurons('EM',
                                       output_filename='EM_predicted_only_pass_tests.html',
                                       only_pass=True)

    with_neurotransmitter.plot_neurons('clem', output_filename='CLEM_predicted_optimize_all_predict.html')
    with_neurotransmitter.plot_neurons('clem', output_filename='CLEM_predicted_only_pass_tests.html', only_pass=True)
