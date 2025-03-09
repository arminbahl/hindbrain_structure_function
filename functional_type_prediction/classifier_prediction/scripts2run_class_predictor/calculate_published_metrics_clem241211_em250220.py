from holoviews.plotting.bokeh.styles import font_size
from sklearn.neighbors import KNeighborsClassifier
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.class_predictor import *
from itertools import product
from sklearn.metrics import f1_score

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
    with_neurotransmitter.calculate_published_metrics()

    n_estimators_rf = 100
    #clf_pv = KNeighborsClassifier()
    #clf_ps = KNeighborsClassifier()
    clf_pv = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_ps = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_ff = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    with_neurotransmitter.confusion_matrices(clf_pv, method='lpo', plot_cm_order_jon=True, feature_type='pv')
    with_neurotransmitter.confusion_matrices(clf_ps, method='lpo', plot_cm_order_jon=True, feature_type='ps')
    with_neurotransmitter.confusion_matrices(clf_ff, method='lpo', plot_cm_order_jon=True, feature_type='ff')
