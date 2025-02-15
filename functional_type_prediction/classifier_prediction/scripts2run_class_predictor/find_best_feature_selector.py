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
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.class_predictor import *

np.set_printoptions(suppress=True)
from hindbrain_structure_function.functional_type_prediction.NBLAST.nblast_matrix_navis import *
from slack_sdk import WebClient
from sklearn.metrics import accuracy_score
from matplotlib.patches import Patch
import scipy.stats as stats


def send_slack_message(RECEIVER="Florian Kämpf", MESSAGE="Script finished!"):
    slack_token = "xoxb-2212881652034-3363495253589-2kSTt6BcH3YTJtb3hIjsOJDp"
    client = WebClient(token=slack_token)
    ul = client.users_list()
    ul['real_name']
    member_list = []

    for users in ul.data["members"]:
        member_list.append(users["profile"]['real_name'])
        if RECEIVER in users["profile"]['real_name']:
            chat_id = users["id"]

    client.conversations_open(users=chat_id)
    response = client.chat_postMessage(
        channel=chat_id,
        text=MESSAGE
    )





if __name__ == "__main__":
    # load metrics and cells
    with_neurotransmitter = class_predictor(Path('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud'))
    with_neurotransmitter.load_cells_df(kmeans_classes=True, new_neurotransmitter=True,
                                        modalities=['pa', 'clem241211', 'em', 'clem_predict241211'], neg_control=True)
    with_neurotransmitter.calculate_metrics('FINAL_CLEM_CLEMPREDICT_EM_with_clem241211_withgregor250215')

    # with_neurotransmitter.calculate_published_metrics()
    with_neurotransmitter.load_cells_features('FINAL_CLEM_CLEMPREDICT_EM_with_clem241211_withgregor250215',
                                              with_neg_control=True,
                                              drop_neurotransmitter=False)
    # throw out truncated, exits and growth cone
    with_neurotransmitter.remove_incomplete()
    # apply gregors manual morphology annotations
    # with_neurotransmitter.add_new_morphology_annotation()
    # select features
    with_neurotransmitter.select_features_RFE('all', 'clem', cv=False, cv_method_RFE='ss',
                                              metric='f1')  # runs through all estimator

    # # without neurotrasnmitter
    # without_neurotransmitter = class_predictor(Path('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud'))
    # without_neurotransmitter.load_cells_df(kmeans_classes=True, new_neurotransmitter=True,
    #                                        modalities=['pa', 'clem', 'em', 'clem_predict'], neg_control=True)
    # without_neurotransmitter.calculate_metrics('FINAL_CLEM_CLEMPREDICT_EM_PA_241204')  #
    # # with_neurotransmitter.calculate_published_metrics()
    # without_neurotransmitter.load_cells_features('FINAL_CLEM_CLEMPREDICT_EM_PA_241204', with_neg_control=True,
    #                                              drop_neurotransmitter=True)
    # # throw out truncated, exits and growth cone
    # without_neurotransmitter.remove_incomplete()
    # # apply gregors manual morphology annotations
    # without_neurotransmitter.add_new_morphology_annotation()
    # without_neurotransmitter.select_features_RFE('all', 'clem', cv=False,
    #                                              cv_method_RFE='ss', metric='f1')  # runs through all estimator
