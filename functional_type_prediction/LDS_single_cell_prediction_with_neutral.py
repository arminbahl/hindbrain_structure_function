import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import copy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
import seaborn as sns
import trimesh as tm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
import matplotlib as mpl

def generate_matching_plot(features,labels,labels_imaging_modality,path,column_labels,solver='lsqr',shrinkage='auto',title='prediction_plot',match_limit=0.5,return_metrics=False):
    #init variables
    prob_matrix = np.empty(shape=(features.shape[0],len(np.unique(labels))))
    pred_matrix = np.empty(shape=(features.shape[0],1),dtype='<U24')
    prediction_correct = []
    legend_elements = []
    used_labels = []
    for label in column_labels:
        used_labels.append(Patch(facecolor='white', edgecolor='white', label=label))

    priors = [len(labels[labels==x])/len(labels) for x in np.unique(labels)]

    #loop over cells
    for i in range(features.shape[0]):
        X_train = features[[x for x in range(features.shape[0]) if x != i]]
        X_test = features[i,:]
        y_train = labels[[x for x in range(features.shape[0]) if x != i]]
        y_test = labels[i]

        #create and fit lda
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage,priors=priors)
        clf.fit(X_train, y_train.flatten())

        # predict
        y_pred = clf.predict(X_test[np.newaxis,:])
        y_prob = clf.predict_proba(X_test[np.newaxis,:])
        prob_matrix[i,:] = y_prob


        if y_prob[0][np.argwhere(clf.classes_==y_pred)[0][0]]>=y_prob[0][np.argwhere(clf.classes_!=y_pred)].sum() and y_prob[0][np.argwhere(clf.classes_==y_pred)[0][0]]>=match_limit:
            prediction_correct.append(y_pred==y_test)
            pred_matrix[i, :] = y_pred

    print(f'\nPredictions correct: {np.round((np.sum(prediction_correct) / features.shape[0] * 100), 2)}%'
          f'\nPredictions incorrect: {np.round((np.sum([not x for x in prediction_correct]) / features.shape[0] * 100), 2)}%'
          f'\nNo prediction: {np.round((features.shape[0] - len(prediction_correct)) / features.shape[0] * 100, 2)}%\n')
    #figure part
    if return_metrics:
        correct = np.round((np.sum(prediction_correct) / features.shape[0] * 100), 2)
        incorrect = np.round((np.sum([not x for x in prediction_correct]) / features.shape[0] * 100), 2)
        no_prediction = np.round((features.shape[0] - len(prediction_correct)) / features.shape[0] * 100, 2)
        return correct, incorrect, no_prediction
    else:
        color_dict_type = {
            "integrator ipsilateral": '#feb326b3',
            "integrator contralateral": '#e84d8ab3',
            "dynamic threshold": '#64c5ebb3',
            "motor command": '#7f58afb3',
            'neg_control': "#a8c256b3"
        }


        color_dict_modality = {'clem': 'black', "photoactivation": "gray"}

        fig, ax = plt.subplots(figsize=(40, 8))

        im = ax.pcolormesh(prob_matrix.T)
        ax.plot([-1, -1], [-1, -1])

        labels_sort = np.unique(labels)
        labels_sort.sort()
        location_dict = {}
        for i, label in enumerate(labels_sort):
            location_dict[label] = i
            ax.plot([-1, -1], [0 + i, 1 + i], color=color_dict_type[label], lw=3, solid_capstyle='butt')
            temp_indices = np.argwhere(labels == label).flatten()
            ax.plot([np.min(temp_indices), np.max(temp_indices) + 1], [-0.25, -0.25], color=color_dict_type[label], lw=3, solid_capstyle='butt', alpha=1)

            if len(np.unique(labels_imaging_modality)) > 1:
                for i2, modality in enumerate(color_dict_modality.keys()):
                    temp_indices = np.argwhere((labels == label) & (labels_imaging_modality == modality))
                    ax.plot([np.min(temp_indices), np.max(temp_indices) + 1], [-0.5, -0.5], color=color_dict_modality[modality], lw=3, solid_capstyle='butt', alpha=1)
                    if not modality in [x.get_label() for x in legend_elements]:
                        legend_elements.append(Patch(facecolor=color_dict_modality[modality], edgecolor=color_dict_modality[modality], label=modality))
            if not label in [x.get_label() for x in legend_elements]:
                legend_elements.append(Patch(facecolor=color_dict_type[label], edgecolor=color_dict_type[label], label=label))


        for x, item in enumerate(pred_matrix):
            if item != '':
                y = location_dict[item[0]]
                plt.plot([x,x],[y,y+1],lw=2,color='red')
                plt.plot([x, x+1], [y+1, y + 1], lw=2, color='red')
                plt.plot([x, x + 1], [y, y ], lw=2, color='red')
                plt.plot([x+1 , x+1], [y, y + 1], lw=2, color='red')

        ax.set_yticks(np.arange(len(labels_sort)) + 0.5, [x + " prediction" for x in labels_sort])
        ax.set_xlim(-2, len(features))
        ax.set_ylim(-2, len(labels_sort))

        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        fig.colorbar(im, orientation='vertical')
        ax.set_xticks([])
        plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.1)
        savepath = path_to_data / 'make_figures_FK_output' / 'LDA_cell_type_prediction'
        os.makedirs(savepath,exist_ok=True)
        os.makedirs(savepath/'png', exist_ok=True)
        os.makedirs(savepath/'pdf', exist_ok=True)
        fig.set_dpi(450)

        first_legend = ax.legend(handles=legend_elements, frameon=False, loc=8, ncol=len(legend_elements))
        ax.add_artist(first_legend)
        second_legend = ax.legend(handles=used_labels, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(1.225, 0.975), alignment='left')
        ax.text(1.195, 1.05, f'Used features N={len(column_labels)}', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, font={'weight': 'heavy', 'size': 20, })
        ax.add_artist(second_legend)
        plt.title(title + f'\nPredictions correct: {np.round((np.sum(prediction_correct) / features.shape[0] * 100), 2)}%'
                          f'\nPredictions incorrect: {np.round((np.sum([not x for x in prediction_correct]) / features.shape[0] * 100), 2)}%'
                          f'\nNo prediction: {np.round((features.shape[0]-len(prediction_correct))/features.shape[0] * 100,2)}%\n')
        plt.savefig(savepath /'pdf'/ (title.replace('\n'," ") + ".pdf"))
        plt.savefig(savepath /'png'/ (title.replace('\n'," ")+ ".png"))

        path_to_open = savepath /'pdf'/ (title.replace('\n'," ")+ ".pdf")

        os.startfile(path_to_open)


def generate_matching_plot_test_and_train_not_the_same(features_train,labels_train,features_test,labels_test,labels_imaging_modality,path,column_labels,solver='lsqr',shrinkage='auto',title='prediction_plot',match_limit=0.5,return_metrics=False):
    #init variables
    prob_matrix = np.empty(shape=(features_test.shape[0], len(np.unique(labels))))
    pred_matrix = np.empty(shape=(features_test.shape[0], 1), dtype='<U24')
    prediction_correct = []
    legend_elements = []
    used_labels = []
    for label in column_labels:
        used_labels.append(Patch(facecolor='white', edgecolor='white', label=label))

    priors = [len(labels[labels == x]) / len(labels) for x in np.unique(labels)]

    # TRAIN
    X_train = features_train
    y_train = labels_train

    # create and fit lda
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors)
    clf.fit(X_train, y_train.flatten())

    # loop over cells
    for i in range(features_test.shape[0]):

        X_test = features_test[i, :]

        y_test = labels_test[i]

        # predict
        y_pred = clf.predict(X_test[np.newaxis, :])
        y_prob = clf.predict_proba(X_test[np.newaxis, :])

        prob_matrix[i, :] = y_prob

        if y_prob[0][np.argwhere(clf.classes_ == y_pred)[0][0]] >= y_prob[0][np.argwhere(clf.classes_ != y_pred)].sum() and y_prob[0][np.argwhere(clf.classes_ == y_pred)[0][0]] >= match_limit:
            prediction_correct.append((y_pred == y_test)[0])
            pred_matrix[i, :] = y_pred

    print(f'\nPredictions correct: {np.round((np.sum(prediction_correct) / features_test.shape[0] * 100), 2)}%'
          f'\nPredictions incorrect: {np.round((np.sum([not x for x in prediction_correct]) / features_test.shape[0] * 100), 2)}%'
          f'\nNo prediction: {np.round((features_test.shape[0] - len(prediction_correct)) / features_test.shape[0] * 100, 2)}%\n')
    # figure part
    if return_metrics:
        correct = np.round((np.sum(prediction_correct) / features_test.shape[0] * 100), 2)
        incorrect = np.round((np.sum([not x for x in prediction_correct]) / features_test.shape[0] * 100), 2)
        no_prediction = np.round((features_test.shape[0] - len(prediction_correct)) / features_test.shape[0] * 100, 2)
        return correct, incorrect, no_prediction
    else:
        color_dict_type = {
            "integrator ipsilateral": '#feb326b3',
            "integrator contralateral": '#e84d8ab3',
            "dynamic threshold": '#64c5ebb3',
            "motor command": '#7f58afb3',
            'neg_control': "#a8c256b3"
        }

        color_dict_modality = {'clem': 'black', "photoactivation": "gray"}

        fig, ax = plt.subplots(figsize=(40, 8))

        im = ax.pcolormesh(prob_matrix.T)
        ax.plot([-1, -1], [-1, -1])

        labels_sort = np.unique(labels_test)
        labels_sort.sort()
        location_dict = {}
        for i, label in enumerate(labels_sort):
            location_dict[label] = i
            ax.plot([-1, -1], [0 + i, 1 + i], color=color_dict_type[label], lw=3, solid_capstyle='butt')
            temp_indices = np.argwhere(labels_test == label).flatten()
            ax.plot([np.min(temp_indices), np.max(temp_indices) + 1], [-0.25, -0.25], color=color_dict_type[label], lw=3, solid_capstyle='butt', alpha=1)

            if len(np.unique(labels_imaging_modality)) > 1:
                for i2, modality in enumerate(color_dict_modality.keys()):
                    temp_indices = np.argwhere((labels_test == label) & (labels_imaging_modality == modality))
                    ax.plot([np.min(temp_indices), np.max(temp_indices) + 1], [-0.5, -0.5], color=color_dict_modality[modality], lw=3, solid_capstyle='butt', alpha=1)
                    if not modality in [x.get_label() for x in legend_elements]:
                        legend_elements.append(Patch(facecolor=color_dict_modality[modality], edgecolor=color_dict_modality[modality], label=modality))
            if not label in [x.get_label() for x in legend_elements]:
                legend_elements.append(Patch(facecolor=color_dict_type[label], edgecolor=color_dict_type[label], label=label))

        for x, item in enumerate(pred_matrix):
            if item != '':
                y = location_dict[item[0]]
                plt.plot([x, x], [y, y + 1], lw=2, color='red')
                plt.plot([x, x + 1], [y + 1, y + 1], lw=2, color='red')
                plt.plot([x, x + 1], [y, y], lw=2, color='red')
                plt.plot([x + 1, x + 1], [y, y + 1], lw=2, color='red')

        ax.set_yticks(np.arange(len(labels_sort)) + 0.5, [x + " prediction" for x in labels_sort])
        ax.set_xlim(-2, len(features_test))
        ax.set_ylim(-2, len(labels_sort))

        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.1)
        fig.colorbar(im, orientation='vertical')
        ax.set_xticks([])

        savepath = path_to_data / 'make_figures_FK_output' / 'LDA_cell_type_prediction'
        os.makedirs(savepath / 'png', exist_ok=True)
        os.makedirs(savepath / 'pdf', exist_ok=True)
        fig.set_dpi(450)
        ax.legend(handles=legend_elements, frameon=False, loc=8, ncols=len(legend_elements))
        first_legend = ax.legend(handles=legend_elements, frameon=False, loc=8, ncol=len(legend_elements))
        ax.add_artist(first_legend)
        second_legend = ax.legend(handles=used_labels, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(1.225, 0.975), alignment='left')
        ax.text(1.195, 1.05, f'Used features N={len(column_labels)}', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, font={'weight': 'heavy', 'size': 20, })
        ax.add_artist(second_legend)
        plt.title(title + f'\nPredictions correct: {np.round((np.sum(prediction_correct) / features_test.shape[0] * 100), 2)}%'
                          f'\nPredictions incorrect: {np.round((np.sum([not x for x in prediction_correct]) / features_test.shape[0] * 100), 2)}%'
                          f'\nNo prediction: {np.round((features_test.shape[0] - len(prediction_correct)) / features_test.shape[0] * 100, 2)}%\n')
        plt.savefig(savepath / 'pdf' / (title.replace('\n', " ") + ".pdf"))
        plt.savefig(savepath / 'png' / (title.replace('\n', " ") + ".png"))

        path_to_open = savepath / 'pdf' / (title.replace('\n', " ") + ".pdf")

        os.startfile(path_to_open)


def single_run(features,labels,feature_labels,random_seed=42,solver='lsqr',shrinkage='auto',test_size=0.3,stratify=True,shuffle_labels=False,output = True):
    #split test train
    X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=test_size, random_state=random_seed)

    # Create the LDA model
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    # Fit the model

    if shuffle_labels:
        clf.fit(X_train,np.random.permutation(y_train.flatten()))
    else:
        if shuffle_labels:
            clf.fit(X_train, y_train.flatten())

    # Make predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)


    #prints
    if output:
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print(f'Accuracy: {round(accuracy * 100, 4)}%')

    return y_pred, y_prob
def determine_important_features(features,labels,feature_labels, repeats=10000,random_seed=42,solver='lsqr',shrinkage='auto',test_size=0.3,stratify=True,return_collection_coef_matrix=False):
    #init variables
    collection_coef_matrix = None
    collection_prediction_correct = []
    #sort features and labels in a predetermined way
    temp_array = np.hstack([features,labels[:,np.newaxis]])
    rng = np.random.default_rng(seed=random_seed)
    temp_array = rng.permutation(temp_array)
    features = temp_array[:,:-1]
    labels = temp_array[:,-1]
    priors = [len(labels[labels == x]) / len(labels) for x in np.unique(labels)]


    for i in tqdm(range(repeats)):
        #split_data
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=test_size, random_state=random_seed)
        else:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_seed)

        # Create the LDA model
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage,priors=priors)
        clf.fit(X_train, y_train.flatten())

        #predict
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        collection_prediction_correct.append(accuracy)

        #weights matrix
        coef_matrix = clf.coef_
        if collection_coef_matrix is None:
            collection_coef_matrix = coef_matrix[np.newaxis, :, :]
        else:
            collection_coef_matrix = np.vstack([collection_coef_matrix, coef_matrix[np.newaxis, :, :]])
        random_seed += 1

    coef_matrix_avg = np.mean(collection_coef_matrix, axis=0)
    features_with_high_weights_bool = np.sum((abs(coef_matrix_avg) > 0.5), axis=0).astype(bool)
    reduced_features_bool = features_with_high_weights_bool
    reduced_features = features[:, features_with_high_weights_bool]
    labels_of_reduced_features = list(np.array(feature_labels)[features_with_high_weights_bool])
    print_message = ''.join([f'- {x}\n' for x in labels_of_reduced_features])

    #print outcome
    print(f"All Features Mean accuracy over {repeats} repeats: {round(np.mean(collection_prediction_correct) * 100, 2)}%")
    print(f"All Features Max accuracy  over {repeats} repeats: {round(np.max(collection_prediction_correct) * 100, 2)}%")
    print(f"All Features Min accuracy over {repeats} repeats: {round(np.min(collection_prediction_correct) * 100, 2)}%")
    print(f"All Features Std of accuracy over {repeats} repeats: {round(np.std(collection_prediction_correct) * 100, 2)}%")
    print(f"\n{len(list(np.array(feature_labels)[features_with_high_weights_bool]))} Features used\n\n{print_message}\n")


    if return_collection_coef_matrix:
        return reduced_features, reduced_features_bool,collection_coef_matrix
    else:
        return reduced_features,reduced_features_bool


def determine_important_features_RFECV(features,labels,feature_labels,solver='lsqr',shrinkage='auto',scoring='accuracy'):
    from sklearn.feature_selection import RFECV
    rfe = RFECV(estimator=LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage), step=1,scoring=scoring,)
    rfe = rfe.fit(features, labels.flatten())
    np.array(feature_labels)[rfe.support_]
    reduced_features = features[:,rfe.support_]
    reduced_features_bool = rfe.support_





    return reduced_features,reduced_features_bool

def determine_important_L1(features, labels):
    from sklearn.svm import LinearSVC
    from sklearn.feature_selection import SelectFromModel

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(features, labels)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(features)
    selected_features_mask = model.get_support()
    return X_new,selected_features_mask

def determine_important_tree(features, labels):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_selection import SelectFromModel

    clf = ExtraTreesClassifier(n_estimators=1000)
    clf = clf.fit(features, labels)
    clf.feature_importances_

    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(features)
    X_new.shape
    selected_features_mask = model.get_support()

    return X_new,selected_features_mask

def determine_important_SFS(features, labels,n_features_to_select = None):
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=3)
    sfs = SequentialFeatureSelector(knn,n_features_to_select=n_features_to_select)
    sfs.fit(features, labels)
    X_new = sfs.transform(features)
    selected_features_mask = sfs.get_support()
    return X_new,selected_features_mask
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # Constants
    repeats = 10000
    path_to_data = get_base_path()



    #load normal cells
    clem_pa_cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem', 'pa'], load_repaired=True)
    prediction_project_cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['prediction_project'],
                                                             load_repaired=True)
    neg_controls = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['neg_controls'],
                                                 load_repaired=True)

    all_cells_w_swc = pd.concat([clem_pa_cells,prediction_project_cells,neg_controls])
    all_cells_w_swc = all_cells_w_swc.drop_duplicates(keep='first', inplace=False,subset='cell_name')
    all_cells_w_swc_no_function =  all_cells_w_swc.loc[(all_cells_w_swc.function == 'nan'), :]
    all_cells_w_swc = all_cells_w_swc.loc[(all_cells_w_swc.function != 'nan'), :]
    all_cells_w_swc = all_cells_w_swc.loc[(~all_cells_w_swc.function.isna()), :]
    all_cells_w_swc = all_cells_w_swc.reset_index(drop=True)
    all_cells_w_swc.loc[all_cells_w_swc['function'].isin(['off-response','no response']),'function'] = 'neg_control'






    #load predict cells
    clem_predict_cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem_predict'], load_repaired=True)
    clem_predict_cells = pd.concat([clem_predict_cells,all_cells_w_swc_no_function])
    clem_predict_cells = clem_predict_cells.drop_duplicates(keep='first', inplace=False, subset='cell_name')
    clem_predict_cells = clem_predict_cells.loc[(~clem_predict_cells['cell_name'].isin(all_cells_w_swc.cell_name)),:]
    clem_predict_cells = clem_predict_cells.loc[clem_predict_cells['type']!='axon',:]
    # all_cells_w_swc.loc[:,'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_w_swc['swc']]
    # all_cells_w_swc.loc[:,'swc'] = [navis.prune_twigs(x, 10, recursive=True) for x in all_cells_w_swc['swc']]
    # clem_predict_cells.loc[:,'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in clem_predict_cells['swc']]
    # clem_predict_cells.loc[:,'swc'] = [navis.prune_twigs(x, 10, recursive=True) for x in clem_predict_cells['swc']]


    from hindbrain_structure_function.functional_type_prediction.LDS_NBLAST_prediction_jon import *
    # Data Loading
    file_path = path_to_data / 'make_figures_FK_output' / 'CLEM_and_PA_features.hdf5'
    fmn = pd.read_hdf(file_path, 'function_morphology_neurotransmitter')
    pp = pd.read_hdf(file_path, 'predictor_pipeline_features')
    ac = pd.read_hdf(file_path, 'angle_cross')

    all_cells = pd.concat([fmn, pp, ac], axis=1)

    file_path = path_to_data / 'make_figures_FK_output' / 'prediction_project_features.hdf5'
    fmn = pd.read_hdf(file_path, 'function_morphology_neurotransmitter')
    pp = pd.read_hdf(file_path, 'predictor_pipeline_features')
    ac = pd.read_hdf(file_path, 'angle_cross')

    all_cells2 = pd.concat([fmn, pp, ac], axis=1)

    all_cells = pd.concat([all_cells, all_cells2])


    #throw out weird jon cells
    all_cells = all_cells.loc[~all_cells.cell_name.isin(["cell_576460752734566521","cell_576460752723528109","cell_576460752684182585"]),:]
    features_train, labels_train, labels_imaging_modality_train, column_labels_train, df_train = load_train_data(path_to_data,file='train_complete')
    labels_train = np.array(['neg_control' if x in ['off-response','no response'] else x for x in labels_train])
    df_train.loc[df_train['function'].isin(['no response', 'off-response ']),'function'] = 'neg_control'
    features_train = features_train[labels_train != 'neg_control',:]
    labels_train = labels_train[labels_train!='neg_control']
    df_train = df_train.loc[~df_train['function'].isin(['no response', 'off-response ']),:]

    #adjust based on regressor
    all_cells_w_swc['prediction_equals_manual'] = False
    all_cells_w_swc['correlation_test_passed'] = False
    all_cells_w_swc['prediction_regressor'] = np.nan
    all_cells_w_swc['prediction_equals_manual_st'] = False
    all_cells_w_swc['correlation_test_passed_st'] = False
    all_cells_w_swc['prediction_regressor_st'] = np.nan

    for i,cell in all_cells_w_swc.iterrows():
        temp_path = Path(str(cell.metadata_path)[:-4] + "_with_regressor.txt")
        temp_path_pa = path_to_data / 'paGFP'/cell.cell_name/f"{cell.cell_name}_metadata_with_regressor.txt"
        if temp_path.exists():
            if cell.imaging_modality == 'photoactivation':
                pass
            with open(temp_path,'r') as f:
                t = f.read()

                all_cells_w_swc.loc[i,'prediction_regressor'] = t.split('\n')[15].split(' ')[2].strip('"')
                all_cells_w_swc.loc[i,'correlation_test_passed'] = eval(t.split('\n')[16].split(' ')[2].strip('"'))
                all_cells_w_swc.loc[i, 'prediction_equals_manual'] = eval(t.split('\n')[17].split(' ')[2].strip('"'))

                all_cells_w_swc.loc[i,'prediction_regressor_st'] = t.split('\n')[18].split(' ')[2].strip('"')
                all_cells_w_swc.loc[i,'correlation_test_passed_st'] = eval(t.split('\n')[19].split(' ')[2].strip('"'))
                all_cells_w_swc.loc[i, 'prediction_equals_manual_st'] = eval(t.split('\n')[20].split(' ')[2].strip('"'))
                all_cells_w_swc.loc[i, 'kmeans_function'] = t.split('\n')[21].split(' ')[2].strip('"')


        elif temp_path_pa.exists():
            with open(temp_path_pa,'r') as f:
                t = f.read()
                all_cells_w_swc.loc[i,'prediction_regressor'] = t.split('\n')[11].split(' ')[2].strip('"')
                all_cells_w_swc.loc[i,'correlation_test_passed'] = eval(t.split('\n')[12].split(' ')[2].strip('"'))
                all_cells_w_swc.loc[i, 'prediction_equals_manual'] = eval(t.split('\n')[13].split(' ')[2].strip('"'))

                all_cells_w_swc.loc[i,'prediction_regressor_st'] = t.split('\n')[14].split(' ')[2].strip('"')
                all_cells_w_swc.loc[i,'correlation_test_passed_st'] = eval(t.split('\n')[15].split(' ')[2].strip('"'))
                all_cells_w_swc.loc[i, 'prediction_equals_manual_st'] = eval(t.split('\n')[16].split(' ')[2].strip('"'))
                try:
                    all_cells_w_swc.loc[i, 'kmeans_function'] = t.split('\n')[17].split(' ')[2].strip('"')
                except:
                    pass

    all_cells_w_swc =all_cells_w_swc.loc[all_cells_w_swc['kmeans_function'] != 'nan', :]
    #df_train = df_train.reset_index(drop=True)


    for i_cont,cell in zip(range(df_train.shape[0]),df_train.iterrows()):
        i,cell = cell
        if cell.cell_name in list(all_cells_w_swc.cell_name):
            temp = all_cells_w_swc.loc[all_cells_w_swc['cell_name']==cell.cell_name,'kmeans_function'].iloc[0]
            if temp == 'integrator':
                df_train.function = temp + " " + cell.morphology_clone
                labels_train[i_cont] = temp + " " + cell.morphology_clone
            else:
                df_train.function = temp.replace('_'," ")
                labels_train[i_cont] = temp.replace('_'," ")








    # # Data Preprocessing
    # without_nan_function = all_cells[all_cells['function'] != 'nan']
    #
    # # Impute NaNs
    # columns_possible_nans = ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross']
    # without_nan_function.loc[:,columns_possible_nans] = without_nan_function[columns_possible_nans].fillna(0)
    #
    #
    #
    # # Function string replacement
    # without_nan_function.loc[:,'function'] = without_nan_function['function'].str.replace('_', ' ')
    # without_nan_function_pa = without_nan_function[without_nan_function['imaging_modality'] == 'photoactivation'].copy()
    # without_nan_function_pa.loc[:,'function'] = without_nan_function_pa['function'].str.replace('_', ' ')
    # without_nan_function_clem = without_nan_function[without_nan_function['imaging_modality'] == 'clem'].copy()
    # without_nan_function_clem.loc[:,'function'] = without_nan_function_clem['function'].str.replace('_', ' ')
    #
    # # Update 'integrator' function
    # def update_integrator(df):
    #     integrator_mask = df['function'] == 'integrator'
    #     df.loc[integrator_mask, 'function'] += " " + df.loc[integrator_mask, 'morphology']
    #
    # update_integrator(without_nan_function)
    # update_integrator(without_nan_function_pa)
    # update_integrator(without_nan_function_clem)
    #
    # # Replace strings with indices
    # columns_replace_string = ['neurotransmitter','morphology']
    # for work_column in columns_replace_string:
    #     for i,unique_feature in enumerate(without_nan_function[work_column].unique()):
    #         without_nan_function.loc[without_nan_function[work_column] == unique_feature, work_column] = i
    #
    # #sort by function an imaging modality
    # without_nan_function = without_nan_function.sort_values(by=['function','morphology','imaging_modality','neurotransmitter'])
    # without_nan_function_pa = without_nan_function_pa.sort_values(by=['function','imaging_modality','morphology','neurotransmitter'])
    # without_nan_function_clem = without_nan_function_clem.sort_values(by=['function','imaging_modality','morphology','neurotransmitter'])
    #
    # # Extract labels
    # labels = without_nan_function['function'].to_numpy()
    # labels_imaging_modality = without_nan_function['imaging_modality'].to_numpy()
    # labels_imaging_modality_pa = without_nan_function_pa['imaging_modality'].to_numpy()
    # labels_imaging_modality_clem = without_nan_function_clem['imaging_modality'].to_numpy()
    # labels_pa = without_nan_function_pa['function'].to_numpy()
    # labels_clem = without_nan_function_clem['function'].to_numpy()
    # column_labels = list(without_nan_function.columns[3:])
    # #column_labels = list(without_nan_function.columns[~without_nan_function.columns.isin(['cell_name',"imaging_modality",'neurotransmitter','function'])])
    #
    # # Extract features
    # features = without_nan_function.iloc[:, 3:].to_numpy()
    # #features = without_nan_function.loc[:,~without_nan_function.columns.isin(['cell_name',"imaging_modality",'neurotransmitter','function'])].to_numpy()
    # features_pa = without_nan_function[without_nan_function['imaging_modality'] == 'photoactivation'].iloc[:, 3:].to_numpy()
    # #features_pa = without_nan_function[without_nan_function['imaging_modality'] == 'photoactivation'].loc[:, ~without_nan_function.columns.isin(['cell_name', "imaging_modality", 'neurotransmitter', 'function'])].to_numpy()
    # features_clem = without_nan_function[without_nan_function['imaging_modality'] == 'clem'].iloc[:, 3:].to_numpy()
    # #features_clem = without_nan_function[without_nan_function['imaging_modality'] == 'clem'].loc[:, ~without_nan_function.columns.isin(['cell_name', "imaging_modality", 'neurotransmitter', 'function'])].to_numpy()
    #
    # # Standardize features
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)
    # features_pa = scaler.transform(features_pa)
    # features_clem = scaler.transform(features_clem)

    features = features_train.astype(float)
    labels = labels_train
    labels_imaging_modality = labels_imaging_modality_train
    column_labels = column_labels_train
    df_train
    # LDA

    reduced_features,reduced_features_index,collection_coef_matrix = determine_important_features(features,labels,column_labels,return_collection_coef_matrix=True)
    reduced_features2, reduced_features_index2 = determine_important_features_RFECV(features, labels, column_labels, scoring='roc_auc_ovo')


    # #BOTH TRAINING AND TESTING
    generate_matching_plot(features,labels,labels_imaging_modality,path=path_to_data,column_labels=np.array(column_labels),title='All features\nTrained on Both\nTested on Both',match_limit=0.5)
    generate_matching_plot(reduced_features2, labels, labels_imaging_modality, path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index2], title='Reduced features\nTrained on Both\nTested on Both\nFeatures selected with RFECV',match_limit=0.5)
    generate_matching_plot(features[:,reduced_features_index],labels,labels_imaging_modality,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],title=f'Reduced features\nTrained on Both\nTested on Both\nFeatures selected with {repeats} repeats',match_limit=0.5)


    # # #BOTH TRAINING AND PA testing
    # generate_matching_plot_test_and_train_not_the_same(features,labels,features_pa,labels_pa,labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels),
    #                                                    title='All features\nTrained on Both\nTested on PA',match_limit=0.5)
    # generate_matching_plot_test_and_train_not_the_same(features[:,reduced_features_index2],labels,features_pa[:,reduced_features_index2],labels_pa,labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index2],
    #                                                    title='Reduced features\nTrained on Both\nTested on PA\nFeatures selected with RFECV',match_limit=0.5)
    # generate_matching_plot_test_and_train_not_the_same(features[:,reduced_features_index],labels,features_pa[:,reduced_features_index],labels_pa,labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
    #                                                     title=f'Reduced features\nTrained on Both\nTested on PA\nFeatures selected with {repeats} repeats',match_limit=0.5)
    #
    # # #BOTH TRAINING AND CLEM testing
    # generate_matching_plot_test_and_train_not_the_same(features,labels,features_clem,labels_clem,labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels),
    #                                                    title='All features\nTrained on Both\nTested on CLEM',match_limit=0.5)
    # generate_matching_plot_test_and_train_not_the_same(features[:,reduced_features_index2],labels,features_clem[:,reduced_features_index2],labels_clem,labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index2],
    #                                                    title='Reduced features\nTrained on Both\nTested on CLEM\nFeatures selected with RFECV',match_limit=0.5)
    # generate_matching_plot_test_and_train_not_the_same(features[:,reduced_features_index],labels,features_clem[:,reduced_features_index],labels_clem,labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
    #                                                     title=f'Reduced features\nTrained on Both\nTested on CLEM\nFeatures selected with {repeats} repeats',match_limit=0.5)


    #
    # #CLEM TRAINING AND PA TESTING
    # generate_matching_plot_test_and_train_not_the_same(features_clem,labels_clem,features_pa,labels_pa,labels_imaging_modality_pa,path=path_to_data,
    #                                                    title='All features\nTrained on CLEM\nTested on PA')
    # generate_matching_plot_test_and_train_not_the_same(features_clem[:,reduced_features_index],labels_clem,features_pa[:,reduced_features_index],labels_pa,labels_imaging_modality_pa,path=path_to_data,
    #                        title='Reduced features\nTrained on CLEM\nTested on PA')
    #
    # #CLEM TRAINING AND CLEM TESTING
    # generate_matching_plot(features_clem,labels_clem,labels_imaging_modality_clem,path=path_to_data,title='All features\nTrained on CLEM\nTested on CLEM')
    # generate_matching_plot(features_clem[:,reduced_features_index],labels_clem,labels_imaging_modality_clem,path=path_to_data,title='Reduced features\nTrained on CLEM\nTested on CLEM')
    #
    # #PA TRAINING AND CLEM TESTING
    # generate_matching_plot_test_and_train_not_the_same(features_pa,labels_pa,features_clem,labels_clem,labels_imaging_modality_clem,path=path_to_data,
    #                                                    title='All features\nTrained on PA\nTested on CLEM')
    # generate_matching_plot_test_and_train_not_the_same(features_pa[:,reduced_features_index],labels_pa,features_clem[:,reduced_features_index],labels_clem,labels_imaging_modality_clem,path=path_to_data,
    #                        title='Reduced features\nTrained on PA\nTested on CLEM')
    #
    # #PA TRAINING AND PA TESTING
    # generate_matching_plot(features_pa,labels_pa,labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels),title='All features\nTrained on PA\nTested on PA',match_limit=0.5)
    # generate_matching_plot(features_pa[:,reduced_features_index],labels_pa,labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],title=f'Reduced features\nTrained on PA\nTested on PA\nFeatures selected with {repeats} repeats',match_limit=0.5)
    # generate_matching_plot(features_pa[:,reduced_features_index2],labels_pa,labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index2],title=f'Reduced features\nTrained on PA\nTested on PA\nFeatures selected with {repeats} repeats',match_limit=0.5)

    # classification_metrics = ['accuracy', 'balanced_accuracy',
    #                           'f1_micro', 'f1_macro', 'f1_weighted',
    #                           'neg_log_loss',
    #                           'precision_micro','precision_macro','precision_weighted',
    #                           'recall_micro','recall_macro','recall_weighted',
    #                           'jaccard_micro','jaccard_macro', 'jaccard_weighted', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted'
    #                           ]
    # for metric in classification_metrics:
    #     try:
    #         rf2,rfi2 = determine_important_features_RFECV(features,labels,column_labels,scoring=metric)
    #         print("\n\033[0m",metric)
    #         generate_matching_plot(features[:, rfi2], labels, labels_imaging_modality, path=path_to_data, title=f'Reduced features\nTrained on Both\nTested on Both\nMetric {metric}')
    #     except Exception as e:
    #         print("\n\033[93m",metric)
    #         print(e,'\033[0m')


    #plot curce sfs across n features



    # correct = {}
    # incorrect = {}
    # no_predict = {}
    # for i in range(features.shape[1]-1):
    #     a_temp,bool_temp = determine_important_SFS(features, labels, n_features_to_select=i+1)
    #
    #     v1,v2,v3 = generate_matching_plot(features[:,bool_temp], labels, labels_imaging_modality, path=path_to_data, column_labels=np.array(column_labels)[bool_temp], title='All features\nTrained on Both\nTested on Both',return_metrics=True)
    #     try:
    #         correct['SFS_both'].append(v1)
    #         incorrect['SFS_both'].append(v2)
    #         no_predict['SFS_both'].append(v3)
    #
    #     except:
    #         correct['SFS_both'] = [v1]
    #         incorrect['SFS_both'] = [v2]
    #         no_predict['SFS_both'] = [v3]
    #
    #     v1,v2,v3 = generate_matching_plot_test_and_train_not_the_same(features[:,bool_temp],labels,features_pa[:,bool_temp],labels_pa,labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[bool_temp],
    #                                                    title='All features\nTrained on Both\nTested on PA',return_metrics=True)
    #     try:
    #         correct['SFS_pa'].append(v1)
    #         incorrect['SFS_pa'].append(v2)
    #         no_predict['SFS_pa'].append(v3)
    #     except:
    #         correct['SFS_pa'] = [v1]
    #         incorrect['SFS_pa'] = [v2]
    #         no_predict['SFS_pa'] = [v3]
    #     v1, v2, v3 = generate_matching_plot_test_and_train_not_the_same(features[:, bool_temp], labels, features_clem[:, bool_temp], labels_clem, labels_imaging_modality_clem, path=path_to_data,
    #                                                                     column_labels=np.array(column_labels)[bool_temp],
    #                                                                     title='All features\nTrained on Both\nTested on CLEM',return_metrics=True)
    #     try:
    #         correct['SFS_clem'].append(v1)
    #         incorrect['SFS_clem'].append(v2)
    #         no_predict['SFS_clem'].append(v3)
    #     except:
    #         correct['SFS_clem'] = [v1]
    #         incorrect['SFS_clem'] = [v2]
    #         no_predict['SFS_clem'] = [v3]
    #
    # #TREE
    # a_temp, bool_temp = determine_important_tree(features, labels)
    #
    # v1, v2, v3 = generate_matching_plot(features[:, bool_temp], labels, labels_imaging_modality, path=path_to_data, column_labels=np.array(column_labels)[bool_temp], title='All features\nTrained on Both\nTested on Both',
    #                                     return_metrics=True)
    # correct['TREE_both'] = [v1]
    # incorrect['TREE_both'] = [v2]
    # no_predict['TREE_both'] = [v3]
    # v1, v2, v3 = generate_matching_plot_test_and_train_not_the_same(features[:, bool_temp], labels, features_pa[:, bool_temp], labels_pa, labels_imaging_modality_pa, path=path_to_data,
    #                                                                 column_labels=np.array(column_labels)[bool_temp],
    #                                                                 title='All features\nTrained on Both\nTested on PA', return_metrics=True)
    # correct['TREE_pa'] = [v1]
    # incorrect['TREE_pa'] = [v2]
    # no_predict['TREE_pa'] = [v3]
    #
    # v1, v2, v3 = generate_matching_plot_test_and_train_not_the_same(features[:, bool_temp], labels, features_clem[:, bool_temp], labels_clem, labels_imaging_modality_clem, path=path_to_data,
    #                                                                 column_labels=np.array(column_labels)[bool_temp],
    #                                                                 title='All features\nTrained on Both\nTested on CLEM', return_metrics=True)
    # correct['TREE_clem'] = [v1]
    # incorrect['TREE_clem'] = [v2]
    # no_predict['TREE_clem'] = [v3]
    # no_predict['TREE_nofeat'] = a_temp.shape[1]
    #
    # #L1
    # a_temp, bool_temp = determine_important_L1(features,labels)
    # v1, v2, v3 = generate_matching_plot(features[:, bool_temp], labels, labels_imaging_modality, path=path_to_data, column_labels=np.array(column_labels)[bool_temp], title='All features\nTrained on Both\nTested on Both',
    #                                     return_metrics=True)
    # correct['L1_both'] = [v1]
    # incorrect['L1_both'] = [v2]
    # no_predict['L1_both'] = [v3]
    # v1, v2, v3 = generate_matching_plot_test_and_train_not_the_same(features[:, bool_temp], labels, features_pa[:, bool_temp], labels_pa, labels_imaging_modality_pa, path=path_to_data,
    #                                                                 column_labels=np.array(column_labels)[bool_temp],
    #                                                                 title='All features\nTrained on Both\nTested on PA', return_metrics=True)
    # correct['L1_pa'] = [v1]
    # incorrect['L1_pa'] = [v2]
    # no_predict['L1_pa'] = [v3]
    #
    # v1, v2, v3 = generate_matching_plot_test_and_train_not_the_same(features[:, bool_temp], labels, features_clem[:, bool_temp], labels_clem, labels_imaging_modality_clem, path=path_to_data,
    #                                                                 column_labels=np.array(column_labels)[bool_temp],
    #                                                                 title='All features\nTrained on Both\nTested on CLEM', return_metrics=True)
    # correct['L1_clem'] = [v1]
    # incorrect['L1_clem'] = [v2]
    # no_predict['L1_clem'] = [v3]
    # no_predict['L1_nofeat'] = a_temp.shape[1]
    #
    # #RFECV
    # a_temp, bool_temp = determine_important_features_RFECV(features,labels, column_labels,scoring='roc_auc_ovo')
    # v1, v2, v3 = generate_matching_plot(features[:, bool_temp], labels, labels_imaging_modality, path=path_to_data, column_labels=np.array(column_labels)[bool_temp], title='All features\nTrained on Both\nTested on Both',
    #                                     return_metrics=True)
    # correct['RFECV_both'] = [v1]
    # incorrect['RFECV_both'] = [v2]
    # no_predict['RFECV_both'] = [v3]
    # v1, v2, v3 = generate_matching_plot_test_and_train_not_the_same(features[:, bool_temp], labels, features_pa[:, bool_temp], labels_pa, labels_imaging_modality_pa, path=path_to_data,
    #                                                                 column_labels=np.array(column_labels)[bool_temp],
    #                                                                 title='All features\nTrained on Both\nTested on PA', return_metrics=True)
    # correct['RFECV_pa'] = [v1]
    # incorrect['RFECV_pa'] = [v2]
    # no_predict['RFECV_pa'] = [v3]
    #
    # v1, v2, v3 = generate_matching_plot_test_and_train_not_the_same(features[:, bool_temp], labels, features_clem[:, bool_temp], labels_clem, labels_imaging_modality_clem, path=path_to_data,
    #                                                                 column_labels=np.array(column_labels)[bool_temp],
    #                                                                 title='All features\nTrained on Both\nTested on CLEM', return_metrics=True)
    # correct['RFECV_clem'] = [v1]
    # incorrect['RFECV_clem'] = [v2]
    # no_predict['RFECV_clem'] = [v3]
    # no_predict['RFECV_nofeat'] = a_temp.shape[1]
    #
    # #repeats
    # v1, v2, v3 =generate_matching_plot(features[:,reduced_features_index], labels, labels_imaging_modality, path=path_to_data, column_labels=np.array(column_labels)[reduced_features_index], title='All features\nTrained on Both\nTested on Both', return_metrics=True)
    # correct['REPEATS_both'] = [v1]
    # incorrect['REPEATS_both'] = [v2]
    # no_predict['REPEATS_both'] = [v3]
    # no_predict['REPEATS_nofeat'] = a_temp.shape[1]
    # v1, v2, v3 =generate_matching_plot_test_and_train_not_the_same(features[:,reduced_features_index], labels, features_pa[:,reduced_features_index], labels_pa, labels_imaging_modality_pa, path=path_to_data, column_labels=np.array(column_labels)[reduced_features_index],
    #                                                    title='All features\nTrained on Both\nTested on PA', return_metrics=True)
    # correct['REPEATS_pa'] = [v1]
    # incorrect['REPEATS_pa'] = [v2]
    # no_predict['REPEATS_pa'] = [v3]
    # v1, v2, v3 =generate_matching_plot_test_and_train_not_the_same(features[:,reduced_features_index], labels, features_clem[:,reduced_features_index], labels_clem, labels_imaging_modality_pa, path=path_to_data, column_labels=np.array(column_labels)[reduced_features_index],
    #                                                    title='All features\nTrained on Both\nTested on CLEM', return_metrics=True)
    # correct['REPEATS_clem'] = [v1]
    # incorrect['REPEATS_clem'] = [v2]
    # no_predict['REPEATS_clem'] = [v3]
    #
    # colordict = {'clem': 'blue', "pa": "red", 'both': "green"}
    #
    # for plot, plot_name in zip([correct, incorrect, no_predict], ['correct', 'incorrect', 'no_predict']):
    #     plt.subplots(figsize=(15, 10))
    #     for modality in ['both', 'clem', 'pa']:
    #         plt.plot(plot["SFS_" + modality], label="SFS_" + modality,c=colordict[modality])
    #         plt.scatter(no_predict['RFECV_nofeat'], plot['RFECV_' + modality], label="RFEC_" + modality,c=colordict[modality],marker='P')
    #         plt.scatter(no_predict['L1_nofeat'], plot['L1_' + modality], label="L1_" + modality,c=colordict[modality],marker='D')
    #         plt.scatter(no_predict['TREE_nofeat'], plot['TREE_' + modality], label="TREE_" + modality,c=colordict[modality],marker='s')
    #         plt.scatter(no_predict['REPEATS_nofeat'], plot['REPEATS_' + modality], label="REPEATS_" + modality,c=colordict[modality],marker='*')
    #
    #     plt.title(plot_name)
    #     plt.legend(loc='best', ncols=2, fontsize='x-small', frameon=False)
    #     plt.show()



    nb = nblast_two_groups(all_cells_w_swc,all_cells_w_swc,shift_neurons=False)
    aaa = navis.nbl.extract_matches(nb, 2)
    nb = nblast_two_groups(all_cells_w_swc,clem_predict_cells,shift_neurons=False)
    bbb = navis.nbl.extract_matches(nb.T, 2)

    cutoff= np.percentile(list(aaa.score_2),10)
    #cutoff = 0.3

    subset_swc = list(bbb.loc[bbb['score_1']>=cutoff,'id'])
    neuron_list = clem_predict_cells.loc[clem_predict_cells['cell_name'].isin(subset_swc),'swc']

    import plotly
    brain_meshes = load_brs(path_to_data, 'raphe')
    fig = navis.plot3d(list(neuron_list), backend='plotly',
                       width=1920, height=1080, hover_name=True, alpha=1)
    fig = navis.plot3d(brain_meshes, backend='plotly', fig=fig,
                       width=1920, height=1080, hover_name=True)
    fig.update_layout(
        scene={
            'xaxis': {'autorange': 'reversed'},  # reverse !!!
            'yaxis': {'autorange': True},

            'zaxis': {'autorange': True},
            'aspectmode': "data",
            'aspectratio': {"x": 1, "y": 1, "z": 1}},
    )
    plotly.offline.plot(fig, filename=f"nothing.html", auto_open=True, auto_play=False)

