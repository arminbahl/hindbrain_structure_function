import chardet
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import LinearSVC
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import plotly

from hindbrain_structure_function.functional_type_prediction.classifier_prediction.calculate_metric2df import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

np.set_printoptions(suppress=True)
from hindbrain_structure_function.functional_type_prediction.NBLAST.nblast_matrix_navis import *
from slack_sdk import WebClient
from sklearn.metrics import accuracy_score
from matplotlib.patches import Patch
import scipy.stats as stats
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.class_predictor import *
from itertools import product
from sklearn.metrics import f1_score

if __name__ == "__main__":
    for drop_nt in [True, False]:
        for acc_or_f1 in ['accuracy', 'f1']:
            for lpo_or_ss, estimator in zip(['lpo', 'ss'], [PassiveAggressiveClassifier(random_state=0),
                                                            RidgeClassifier(random_state=0)]):
                for em_input in [True, False]:
                    # load metrics and cells
                    with_neurotransmitter = class_predictor(
                        Path('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud'))
                    with_neurotransmitter.load_cells_df(kmeans_classes=True, new_neurotransmitter=True,
                                                        modalities=['pa', 'clem', 'em', 'clem_predict'],
                                                        neg_control=True, input_em=em_input)

                    with_neurotransmitter.calculate_metrics('FINAL_CLEM_CLEMPREDICT_EM_PA_241204')
                    with_neurotransmitter.load_cells_features('FINAL_CLEM_CLEMPREDICT_EM_PA_241204',
                                                              with_neg_control=True,
                                                              drop_neurotransmitter=drop_nt)
                    # throw out truncated, exits and growth cone
                    with_neurotransmitter.remove_incomplete()
                    # apply gregors manual morphology annotations
                    with_neurotransmitter.add_new_morphology_annotation()

                    with_neurotransmitter.select_features_RFE('all', 'clem', cv=False, save_features=True,
                                                              estimator=estimator, cv_method_RFE=lpo_or_ss,
                                                              metric=acc_or_f1)  # RidgeClassifier(random_state=0) Perceptron(random_state=0)

                    # predict cells
                    with_neurotransmitter.predict_cells(use_jon_priors=False, suffix='_optimize_all_predict')
                    with_neurotransmitter.calculate_verification_metrics(calculate_smat=False, with_kunst=False)

                    print('no jon prior\n___________________')
                    print('Drop NT:', drop_nt, '\nACC or F1', acc_or_f1, '\nLPO or SS', lpo_or_ss, '\nEM input',
                          em_input, '\n___________________\n')

                    print(with_neurotransmitter.prediction_predict_df.loc[
                              with_neurotransmitter.prediction_predict_df['cell_name'].isin(['147009', '102596']), [
                                  'cell_name', 'prediction', 'prediction_scaled']])
                    acc, _ = with_neurotransmitter.do_cv(method='lpo', clf=LinearDiscriminantAnalysis(solver='lsqr',
                                                                                                      shrinkage='auto'),
                                                         feature_type='fk', train_mod='all', test_mod='clem')
                    f1, _ = with_neurotransmitter.do_cv(method='lpo',
                                                        clf=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                                                        feature_type='fk', train_mod='all', test_mod='clem',
                                                        metric='f1')
                    print('___________________\nF1', f1, '\n')
                    print('Accuracy', acc, '\n___________________\n\n')
                    import time

                    time.sleep(10)
                    print('jon prior')
                    with_neurotransmitter.predict_cells(use_jon_priors=True, suffix='_optimize_all_predict')
                    with_neurotransmitter.calculate_verification_metrics(calculate_smat=False, with_kunst=False)
                    print(with_neurotransmitter.prediction_predict_df.loc[
                              with_neurotransmitter.prediction_predict_df['cell_name'].isin(['147009', '102596']), [
                                  'cell_name', 'prediction', 'prediction_scaled']])

                    acc, _ = with_neurotransmitter.do_cv(method='lpo', clf=LinearDiscriminantAnalysis(solver='lsqr',
                                                                                                      shrinkage='auto'),
                                                         feature_type='fk', train_mod='all', test_mod='clem')
                    f1, _ = with_neurotransmitter.do_cv(method='lpo',
                                                        clf=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                                                        feature_type='fk', train_mod='all', test_mod='clem',
                                                        metric='f1')
                    print('F1', f1)
                    print('Accuracy', acc, '\n')
