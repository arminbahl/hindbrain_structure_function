from hindbrain_structure_function.functional_type_prediction.classifier_prediction.class_predictor import *

if __name__ == "__main__":
    # load metrics and cells
    with_neurotransmitter = class_predictor(Path('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud'))
    with_neurotransmitter.load_cells_df(kmeans_classes=True, new_neurotransmitter=True,
                                        modalities=['pa', 'clem', 'em', 'clem_predict'], neg_control=True)
    with_neurotransmitter.calculate_metrics('FINAL_CLEM_CLEMPREDICT_EM_PA')

    # with_neurotransmitter.calculate_published_metrics()
    with_neurotransmitter.load_cells_features('FINAL_CLEM_CLEMPREDICT_EM_PA', with_neg_control=True,
                                              drop_neurotransmitter=False)
    # throw out truncated, exits and growth cone
    with_neurotransmitter.remove_incomplete()
    # apply gregors manual morphology annotations
    with_neurotransmitter.add_new_morphology_annotation()
    # select features
    with_neurotransmitter.select_features_RFE('all', 'clem', cv=False, save_features=True,
                                              estimator=LogisticRegression(random_state=0), cv_method_RFE='lpo')
    # select classifiers for the confusion matrices
    clf_fk = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    # make confusion matrices
    with_neurotransmitter.confusion_matrices(clf_fk, method='lpo')
    # predict cells
    with_neurotransmitter.predict_cells(use_jon_priors=False, suffix='_optimize_all_predict', predict_recorded=True)
    with_neurotransmitter.plot_neurons('EM', output_filename='EM_predicted_optimize_all_predict.html')
    with_neurotransmitter.plot_neurons('clem', output_filename='CLEM_predicted_optimize_all_predict.html')
    with_neurotransmitter.calculate_verification_metrics(calculate_smat=False, with_kunst=False,
                                                         calculate4recorded=False)
