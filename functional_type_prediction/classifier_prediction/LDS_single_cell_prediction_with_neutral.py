import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.patches import Patch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.calculate_metric2df import *
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel

def plot_prediction_matrix(features,labels,labels_imaging_modality,path,column_labels,solver='lsqr',shrinkage='auto',title='prediction_plot',match_limit=0.5,return_metrics=False):
    #init variables
    prob_matrix = np.empty(shape=(features.shape[0],len(np.unique(labels))))
    pred_matrix = np.empty(shape=(features.shape[0],1),dtype='<U24')
    pred_correct = np.empty(shape=(features.shape[0], 1))
    no_predict = np.empty(shape=(features.shape[0], 1))

    used_features= []
    for label in column_labels:
        used_features.append(Patch(facecolor='white', edgecolor='white', label=label))
    priors = [len(labels[labels == x]) / len(labels) for x in np.unique(labels)]
    #predict all cells one by one
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
            pred_correct[i] = y_pred==y_test
            pred_matrix[i] = y_pred
            no_predict[i] = False
        else:
            pred_correct[i] = np.nan
            pred_matrix[i] = np.nan
            no_predict[i] = True

    pred_correct = pred_correct.flatten()

    n_predictions_correct= np.sum(pred_correct[~np.isnan(pred_correct)])
    n_predictions_incorrect = np.sum([not x for x in pred_correct[~np.isnan(pred_correct)]])
    n_no_predicitons = np.sum(no_predict[~np.isnan(no_predict)])

    convert_percent = lambda x: round((x/features.shape[0])*100,2)
    convert_percent_specific = lambda x,n: round((x / n) * 100, 2)

    print(f'\nPredictions correct: {convert_percent(n_predictions_correct)}%'
          f'\nPredictions incorrect: {convert_percent(n_predictions_incorrect)}%'
          f'\nNo prediction: {convert_percent(n_no_predicitons)}%\n')
    percent_correct_per_class_dict = {}
    n_correct_per_class_dict = {}
    for unique_label in np.unique(labels):
        n_correct_per_class_dict[unique_label] = np.sum(pred_matrix[np.where(labels == unique_label)] == unique_label)
        percent_correct_per_class_dict[unique_label] = convert_percent_specific(n_correct_per_class_dict[unique_label], np.sum(labels == unique_label))
    if return_metrics:



        return n_predictions_correct, n_predictions_incorrect, n_no_predicitons,n_correct_per_class_dict,percent_correct_per_class_dict
    else:
        color_dict_type = {
            "integrator ipsilateral": '#feb326b3',
            "integrator contralateral": '#e84d8ab3',
            "dynamic threshold": '#64c5ebb3',
            "motor command": '#7f58afb3',
            'neg control': "#a8c256b3"
        }

        color_dict_modality = {'clem': 'black', "photoactivation": "gray"}



        fig, ax = plt.subplots(figsize=(40, 8))
        labels_sort = np.unique(labels)
        labels_sort.sort()
        legend_elements = []

        im = ax.pcolormesh(prob_matrix.T)

        fig.colorbar(im, orientation='vertical')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.1)
        savepath = path_to_data / 'make_figures_FK_output' / 'LDA_cell_type_prediction'
        os.makedirs(savepath,exist_ok=True)
        os.makedirs(savepath/'png', exist_ok=True)
        os.makedirs(savepath/'pdf', exist_ok=True)


        # fig.set_dpi(300)

        #plot functional class indicator
        def intersection(list_a, list_b):
            list_a = list_a[0]
            list_b = list_b[0]
            return [e for e in list_a if e in list_b]
        for unique_class in np.unique(labels):
            for unique_modality in np.unique(labels_imaging_modality[np.where(labels == unique_class)]):
                intersect = intersection(np.where(labels_imaging_modality == unique_modality),np.where(labels == unique_class))
                min_index2 = np.min(intersect)
                max_index2 = np.max(intersect)
                plt.plot([min_index2, max_index2 + 1], [-0.75, -0.75], c=color_dict_modality[unique_modality],lw=3, solid_capstyle='butt')
                if not unique_modality in [x.get_label() for x in legend_elements]:
                    legend_elements.append(Patch(facecolor=color_dict_modality[unique_modality], edgecolor=color_dict_modality[unique_modality], label=unique_modality))

            min_index = np.min(np.where(labels == unique_class))
            max_index = np.max(np.where(labels == unique_class))


            ax.text((max_index+min_index)/2,-0.25,f"{n_correct_per_class_dict[unique_class]}/{np.sum(labels == unique_class)}",font={'size': 10, },horizontalalignment='center',verticalalignment='center')

            plt.plot([min_index,max_index+1],[-0.5,-0.5],c=color_dict_type[unique_class],lw=3, solid_capstyle='butt')
            if not unique_class in [x.get_label() for x in legend_elements]:
                legend_elements.append(Patch(facecolor=color_dict_type[unique_class], edgecolor=color_dict_type[unique_class], label=unique_class))



        for unique_class,i in zip(np.unique(labels),range(len(np.unique(labels)))):
            plt.plot([-1,-1],[i,i+1],c=color_dict_type[unique_class],lw=3, solid_capstyle='butt')

        first_legend = ax.legend(handles=legend_elements, frameon=False, loc=8, ncol=len(legend_elements))

        for x, item in enumerate(pred_matrix):
            if item != 'nan':
                y = np.argwhere(labels_sort==item[0]).flatten()[0]
                plt.plot([x,x],[y,y+1],lw=2,color='red')
                plt.plot([x, x+1], [y+1, y + 1], lw=2, color='red')
                plt.plot([x, x + 1], [y, y ], lw=2, color='red')
                plt.plot([x+1 , x+1], [y, y + 1], lw=2, color='red')
        used_labels = []
        for label in column_labels:
            used_labels.append(Patch(facecolor='white', edgecolor='white', label=label))
        second_legend = ax.legend(handles=used_labels, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(1.225, 0.975), alignment='left')

        ax.plot([-1, -1], [-1, -1])
        ax.set_yticks(np.arange(len(labels_sort)) + 0.5, [x + " prediction" for x in labels_sort])
        ax.set_xlim(-2, len(features))
        ax.set_ylim(-2, len(labels_sort))
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(labels_sort)) + 0.5, [x + " prediction" for x in labels_sort])
        ax.set_xlim(-2, len(features))
        ax.set_ylim(-2, len(labels_sort))
        ax.add_artist(first_legend)
        ax.text(1.195, 1.05, f'Used features N={len(column_labels)}', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, font={'weight': 'heavy', 'size': 20, })
        ax.add_artist(second_legend)

        plt.title(title +   f'\nPredictions correct: {convert_percent(n_predictions_correct)}%'
                            f'\nPredictions incorrect: {convert_percent(n_predictions_incorrect)}%'
                            f'\nNo prediction: {convert_percent(n_no_predicitons)}%\n')
        plt.savefig(savepath /'pdf'/ (title.replace('\n'," ") + ".pdf"))
        plt.savefig(savepath /'png'/ (title.replace('\n'," ")+ ".png"))

        path_to_open = savepath /'pdf'/ (title.replace('\n'," ")+ ".pdf")

        os.startfile(path_to_open)

        plt.show()
def plot_prediction_matrix_test_train_different(features_train,labels_train,features_test,labels_test,labels_imaging_modality,path,column_labels,train_mod_pa=False,solver='lsqr',shrinkage='auto',title='prediction_plot',match_limit=0.5,return_metrics=False):
    #init variables
    features = features_test
    labels = labels_test

    if train_mod_pa:
        features = features[np.where(labels != 'neg control')]
        labels = labels[np.where(labels != 'neg control')]

    train_contains_test = np.any(np.all(features_train[:, None] == features, axis=2), axis=1).any()

    prob_matrix = np.empty(shape=(features.shape[0],len(np.unique(labels_train))))
    pred_matrix = np.empty(shape=(features.shape[0],1),dtype='<U24')
    pred_correct = np.empty(shape=(features.shape[0], 1))
    no_predict = np.empty(shape=(features.shape[0], 1))

    used_features= []
    for label in column_labels:
        used_features.append(Patch(facecolor='white', edgecolor='white', label=label))


    if not train_contains_test:
        priors = [len(labels[labels == x]) / len(labels) for x in np.unique(labels_train)]
        #predict all cells one by one
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors)
        clf.fit(features_train, labels_train.flatten())

        for i in range(features.shape[0]):
            X_test = features[i,:]
            y_test = labels[i]

            # predict
            y_pred = clf.predict(X_test[np.newaxis,:])
            y_prob = clf.predict_proba(X_test[np.newaxis,:])
            prob_matrix[i,:] = y_prob


            if y_prob[0][np.argwhere(clf.classes_==y_pred)[0][0]]>=y_prob[0][np.argwhere(clf.classes_!=y_pred)].sum() and y_prob[0][np.argwhere(clf.classes_==y_pred)[0][0]]>=match_limit:
                pred_correct[i] = y_pred==y_test
                pred_matrix[i] = y_pred
                no_predict[i] = False
            else:
                pred_correct[i] = np.nan
                pred_matrix[i] = np.nan
                no_predict[i] = True
    elif train_contains_test:
        for i in range(features.shape[0]):

            idx_test_in_train = np.argmax(np.any(np.all(features_train[:, None] == features_test[i, :], axis=2), axis=1))
            X_train = features_train[[x for x in range(features_train.shape[0]) if x != idx_test_in_train]]
            y_train = labels_train[[x for x in range(features_train.shape[0]) if x != idx_test_in_train]]

            priors = [len(y_train[y_train == x]) / len(y_train) for x in np.unique(y_train)]
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
            clf.fit(X_train, y_train.flatten())

            X_test = features_test[i, :]
            y_test = labels_test[i]

            # predict
            y_pred = clf.predict(X_test[np.newaxis, :])
            y_prob = clf.predict_proba(X_test[np.newaxis, :])
            prob_matrix[i, :] = y_prob

            if y_prob[0][np.argwhere(clf.classes_ == y_pred)[0][0]] >= y_prob[0][np.argwhere(clf.classes_ != y_pred)].sum() and y_prob[0][np.argwhere(clf.classes_ == y_pred)[0][0]] >= match_limit:
                pred_correct[i] = y_pred == y_test
                pred_matrix[i] = y_pred
                no_predict[i] = False
            else:
                pred_correct[i] = np.nan
                pred_matrix[i] = np.nan
                no_predict[i] = True


    pred_correct = pred_correct.flatten()

    n_predictions_correct= np.sum(pred_correct[~np.isnan(pred_correct)])
    n_predictions_incorrect = np.sum([not x for x in pred_correct[~np.isnan(pred_correct)]])
    n_no_predicitons = np.sum(no_predict[~np.isnan(no_predict)])

    convert_percent = lambda x: round((x/features.shape[0])*100,2)
    convert_percent_specific = lambda x,n: round((x / n) * 100, 2)

    print(f'\nPredictions correct: {convert_percent(n_predictions_correct)}%'
          f'\nPredictions incorrect: {convert_percent(n_predictions_incorrect)}%'
          f'\nNo prediction: {convert_percent(n_no_predicitons)}%\n')
    percent_correct_per_class_dict = {}
    n_correct_per_class_dict = {}
    for unique_label in np.unique(labels):
        n_correct_per_class_dict[unique_label] = np.sum(pred_matrix[np.where(labels == unique_label)] == unique_label)
        percent_correct_per_class_dict[unique_label] = convert_percent_specific(n_correct_per_class_dict[unique_label], np.sum(labels == unique_label))
    if return_metrics:



        return n_predictions_correct, n_predictions_incorrect, n_no_predicitons,n_correct_per_class_dict,percent_correct_per_class_dict
    else:
        color_dict_type = {
            "integrator ipsilateral": '#feb326b3',
            "integrator contralateral": '#e84d8ab3',
            "dynamic threshold": '#64c5ebb3',
            "motor command": '#7f58afb3',
            'neg control': "#a8c256b3"
        }

        color_dict_modality = {'clem': 'black', "photoactivation": "gray"}



        fig, ax = plt.subplots(figsize=(40, 8))
        labels_sort = np.unique(labels_train)
        labels_sort.sort()
        legend_elements = []

        im = ax.pcolormesh(prob_matrix.T)

        fig.colorbar(im, orientation='vertical')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.1)
        savepath = path_to_data / 'make_figures_FK_output' / 'LDA_cell_type_prediction'
        os.makedirs(savepath,exist_ok=True)
        os.makedirs(savepath/'png', exist_ok=True)
        os.makedirs(savepath/'pdf', exist_ok=True)


        # fig.set_dpi(300)

        #plot functional class indicator
        def intersection(list_a, list_b):
            list_a = list_a[0]
            list_b = list_b[0]
            return [e for e in list_a if e in list_b]
        for unique_class in np.unique(labels):
            for unique_modality in np.unique(labels_imaging_modality[np.where(labels == unique_class)]):
                intersect = intersection(np.where(labels_imaging_modality == unique_modality),np.where(labels == unique_class))
                min_index2 = np.min(intersect)
                max_index2 = np.max(intersect)
                plt.plot([min_index2, max_index2 + 1], [-0.75, -0.75], c=color_dict_modality[unique_modality],lw=3, solid_capstyle='butt')
                if not unique_modality in [x.get_label() for x in legend_elements]:
                    legend_elements.append(Patch(facecolor=color_dict_modality[unique_modality], edgecolor=color_dict_modality[unique_modality], label=unique_modality))

            min_index = np.min(np.where(labels == unique_class))
            max_index = np.max(np.where(labels == unique_class))


            ax.text((max_index+min_index)/2,-0.25,f"{n_correct_per_class_dict[unique_class]}/{np.sum(labels == unique_class)}",font={'size': 10, },horizontalalignment='center',verticalalignment='center')

            plt.plot([min_index,max_index+1],[-0.5,-0.5],c=color_dict_type[unique_class],lw=3, solid_capstyle='butt')
            if not unique_class in [x.get_label() for x in legend_elements]:
                legend_elements.append(Patch(facecolor=color_dict_type[unique_class], edgecolor=color_dict_type[unique_class], label=unique_class))

        for unique_class in np.unique(labels):

            if not unique_class in [x.get_label() for x in legend_elements]:
                legend_elements.append(Patch(facecolor=color_dict_type[unique_class], edgecolor=color_dict_type[unique_class], label=unique_class))



        for unique_class,i in zip(np.unique(labels_train),range(len(np.unique(labels_train)))):
            plt.plot([-1,-1],[i,i+1],c=color_dict_type[unique_class],lw=3, solid_capstyle='butt')

        first_legend = ax.legend(handles=legend_elements, frameon=False, loc=8, ncol=len(legend_elements))

        for x, item in enumerate(pred_matrix):
            if item != 'nan':
                y = np.argwhere(labels_sort==item[0]).flatten()[0]
                plt.plot([x,x],[y,y+1],lw=2,color='red')
                plt.plot([x, x+1], [y+1, y + 1], lw=2, color='red')
                plt.plot([x, x + 1], [y, y ], lw=2, color='red')
                plt.plot([x+1 , x+1], [y, y + 1], lw=2, color='red')
        used_labels = []
        for label in column_labels:
            used_labels.append(Patch(facecolor='white', edgecolor='white', label=label))
        second_legend = ax.legend(handles=used_labels, frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(1.225, 0.975), alignment='left')

        ax.plot([-1, -1], [-1, -1])
        ax.set_yticks(np.arange(len(labels_sort)) + 0.5, [x + " prediction" for x in labels_sort])
        ax.set_xlim(-2, len(features))
        ax.set_ylim(-2, len(labels_sort))
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(labels_sort)) + 0.5, [x + " prediction" for x in labels_sort])
        ax.set_xlim(-2, len(features))
        ax.set_ylim(-2, len(labels_sort))
        ax.add_artist(first_legend)
        ax.text(1.195, 1.05, f'Used features N={len(column_labels)}', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, font={'weight': 'heavy', 'size': 20, })
        ax.add_artist(second_legend)

        plt.title(title +   f'\nPredictions correct: {convert_percent(n_predictions_correct)}%'
                            f'\nPredictions incorrect: {convert_percent(n_predictions_incorrect)}%'
                            f'\nNo prediction: {convert_percent(n_no_predicitons)}%\n')
        plt.savefig(savepath /'pdf'/ (title.replace('\n'," ") + ".pdf"))
        plt.savefig(savepath /'png'/ (title.replace('\n'," ")+ ".png"))

        path_to_open = savepath /'pdf'/ (title.replace('\n'," ")+ ".pdf")

        os.startfile(path_to_open)

        plt.show()
def determine_important_features(features,labels,feature_labels, repeats=10000,random_seed=42,solver='lsqr',shrinkage='auto',test_size=0.3,stratify=True,return_collection_coef_matrix=False,value_automatic_lim=80,per_class_selection=None):
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
    if per_class_selection is not None:
        abs_coef_matrix = np.abs(np.mean(collection_coef_matrix,axis=0))
        best_value_matrix = np.full_like(abs_coef_matrix,0)
        for i in range(abs_coef_matrix.shape[1]):
            best_value_matrix[np.argmax(abs_coef_matrix[:,i]),i] = np.max(abs_coef_matrix[:,i])

        plt.pcolormesh(best_value_matrix)
        plt.colorbar()
        plt.show()
        selected_features = np.array([])
        for i in range(abs_coef_matrix.shape[0]):
            aaa = np.argsort(best_value_matrix[i, :])[::-1][:per_class_selection]
            selected_features = np.concatenate([selected_features,aaa]).astype(int)
        features_with_high_weights_bool = np.full((abs_coef_matrix.shape[1]), False)
        features_with_high_weights_bool[selected_features] = True
    elif value_automatic_lim is not None:
        test_lim = np.percentile(np.max(abs(coef_matrix_avg), axis=0), value_automatic_lim)
        features_with_high_weights_bool = np.sum((abs(coef_matrix_avg) > test_lim), axis=0).astype(bool)



    else:
        features_with_high_weights_bool = np.sum((abs(coef_matrix_avg) > 0.9), axis=0).astype(bool)
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

    rfe = RFECV(estimator=LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage), step=1,scoring=scoring,)
    rfe = rfe.fit(features, labels.flatten())
    np.array(feature_labels)[rfe.support_]
    reduced_features = features[:,rfe.support_]
    reduced_features_bool = rfe.support_





    return reduced_features,reduced_features_bool
def find_optimum_SKB(features_train,labels_train,features_test,labels_test,train_test_identical,train_contains_test,train_mod):
    pred_correct_dict_over_n = {}
    pred_correct_dict_over_n_per_class = {}
    used_features_idx_over_n = {}
    proba_matrix_over_n = {}
    if train_test_identical:
        for evaluator, evaluator_name in zip([f_classif, mutual_info_classif], ['f_classif', 'mutual_info_classif']):
            pred_correct_dict_over_n[evaluator_name] = []
            pred_correct_dict_over_n_per_class[evaluator_name] = []
            used_features_idx_over_n[evaluator_name] = {}
            for no_features in range(1, features_train.shape[1] + 1):
                np.random.seed(42)

                SKB = SelectKBest(evaluator, k=no_features).fit(features_train, labels_train)
                idx = SKB.get_support()

                used_features_idx_over_n[evaluator_name][no_features] = idx
                pred_correct_list = []
                X_new = features_train[:, idx]
                for i in range(X_new.shape[0]):
                    X_train = X_new[[x for x in range(X_new.shape[0]) if x != i]]
                    X_test = X_new[i, :]
                    y_train = labels_train[[x for x in range(X_new.shape[0]) if x != i]]
                    y_test = labels_train[i]

                    priors = [len(y_train[y_train == x]) / len(y_train) for x in np.unique(y_train)]

                    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                    clf.fit(X_train, y_train.flatten())


                    if np.max(clf.predict_proba(X_test[np.newaxis, :]))>= 0.5:
                        pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                    else:
                        pred_correct_list.append(None)


                pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

                temp_list = []
                for unique_label in np.unique(labels_test):
                    correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test==unique_label)] if x is not None])
                    percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test==unique_label)])
                    temp_list.append(correct_in_class/percent_correct_in_class)
                pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))
        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
    elif train_contains_test:

        if train_mod.lower() == 'pa':
            features_test = features_test[np.where(labels_test != 'neg control')]
            labels_test = labels_test[np.where(labels_test != 'neg control')]

        for evaluator, evaluator_name in zip([f_classif, mutual_info_classif], ['f_classif', 'mutual_info_classif']):
            pred_correct_dict_over_n[evaluator_name] = []
            pred_correct_dict_over_n_per_class[evaluator_name] = []
            used_features_idx_over_n[evaluator_name] = {}
            for no_features in range(1, features_train.shape[1] + 1):
                np.random.seed(42)
                SKB = SelectKBest(evaluator, k=no_features).fit(features_train, labels_train)
                idx = SKB.get_support()

                used_features_idx_over_n[evaluator_name][no_features] = idx
                pred_correct_list = []

                X_new = features_train[:,idx]
                for i3 in range(features_test.shape[0]):
                    idx_test_in_train = np.argmax(np.any(np.all(X_new[:, None] == features_test[i3, idx], axis=2), axis=1))
                    features_train_without_test = X_new[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]
                    labels_train_without_test = labels_train[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]

                    priors = [len(labels_train_without_test[labels_train_without_test == x]) / len(labels_train_without_test) for x in np.unique(labels_train_without_test)]
                    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                    clf.fit(features_train_without_test, labels_train_without_test.flatten())

                    X_test = features_test[i3, idx]
                    y_test = labels_test[i3]
                    if np.max(clf.predict_proba(X_test[np.newaxis, :]))>= 0.5:
                        pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                    else:
                        pred_correct_list.append(None)
                pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

                temp_list = []
                for unique_label in np.unique(labels_test):
                    correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test==unique_label)] if x is not None])
                    percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test==unique_label)])
                    temp_list.append(correct_in_class/percent_correct_in_class)
                pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))

        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
    else:
        for evaluator, evaluator_name in zip([f_classif, mutual_info_classif], ['f_classif', 'mutual_info_classif']):
            pred_correct_dict_over_n[evaluator_name] = []
            pred_correct_dict_over_n_per_class[evaluator_name] = []
            used_features_idx_over_n[evaluator_name] = {}
            if train_mod.lower() == 'pa':
                features_test = features_test[np.where(labels_test != 'neg control')]
                labels_test = labels_test[np.where(labels_test != 'neg control')]

            for no_features in range(1, features_train.shape[1] + 1):
                np.random.seed(42)
                SKB = SelectKBest(evaluator, k=no_features)
                X_new = SKB.fit_transform(features_train, labels_train)
                idx = SKB.get_support()

                used_features_idx_over_n[evaluator_name][no_features] = idx
                pred_correct_list = []

                priors = [len(labels_train[labels_train == x]) / len(labels_train) for x in np.unique(labels_train)]
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                clf.fit(X_new, labels_train.flatten())
                for i3 in range(features_test.shape[0]):
                    X_test = features_test[i3, idx]
                    y_test = labels_test[i3]


                    if np.max(clf.predict_proba(X_test[np.newaxis, :]))>= 0.5:
                        pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                    else:
                        pred_correct_list.append(None)


                pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

                temp_list = []
                for unique_label in np.unique(labels_test):
                    correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test==unique_label)] if x is not None])
                    percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test==unique_label)])
                    temp_list.append(correct_in_class/percent_correct_in_class)
                pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))
        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
def find_optimum_PI(features_train,labels_train,features_test,labels_test,train_test_identical,train_contains_test,train_mod):
    evaluator_name = 'permutation importance'

    pred_correct_dict_over_n = {}
    pred_correct_dict_over_n_per_class = {}
    used_features_idx_over_n = {}
    proba_matrix_over_n = {}
    if train_test_identical:
        np.random.seed(42)

        X, y = features_train, labels_train

        unique_strings = list(set(y))
        string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
        y = [string_to_int[string] for string in y]

        model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
        results = permutation_importance(model, X, y, scoring='accuracy', n_repeats=1000)
        importance = results.importances_mean
        pred_correct_dict_over_n[evaluator_name] = []
        pred_correct_dict_over_n_per_class[evaluator_name] = []
        used_features_idx_over_n[evaluator_name] = {}


        for no_features in range(1, features_train.shape[1] + 1):


            idx = importance>=importance[np.argsort(importance, axis=0)[::-1][no_features-1]]
            
            used_features_idx_over_n[evaluator_name][no_features] = idx
            pred_correct_list = []
            X_new = features_train[:, idx]
            for i in range(X_new.shape[0]):
                X_train = X_new[[x for x in range(X_new.shape[0]) if x != i]]
                X_test = X_new[i, :]
                y_train = labels_train[[x for x in range(X_new.shape[0]) if x != i]]
                y_test = labels_train[i]

                priors = [len(y_train[y_train == x]) / len(y_train) for x in np.unique(y_train)]

                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                clf.fit(X_train, y_train.flatten())


                if np.max(clf.predict_proba(X_test[np.newaxis, :]))>= 0.5:
                    pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                else:
                    pred_correct_list.append(None)


            pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

            temp_list = []
            for unique_label in np.unique(labels_test):
                correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test==unique_label)] if x is not None])
                percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test==unique_label)])
                temp_list.append(correct_in_class/percent_correct_in_class)
            pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))
        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
    elif train_contains_test:

        if train_mod.lower() == 'pa':
            features_test = features_test[np.where(labels_test != 'neg control')]
            labels_test = labels_test[np.where(labels_test != 'neg control')]

        X, y = features_train, labels_train

        unique_strings = list(set(y))
        string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
        y = [string_to_int[string] for string in y]

        model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
        results = permutation_importance(model, X, y, scoring='accuracy', n_repeats=1000)
        importance = results.importances_mean



        pred_correct_dict_over_n[evaluator_name] = []
        pred_correct_dict_over_n_per_class[evaluator_name] = []
        used_features_idx_over_n[evaluator_name] = {}
        for no_features in range(1, features_train.shape[1] + 1):
            np.random.seed(42)
            idx = importance>=importance[np.argsort(importance, axis=0)[::-1][no_features-1]]
            

            used_features_idx_over_n[evaluator_name][no_features] = idx
            pred_correct_list = []

            X_new = features_train[:,idx]
            for i3 in range(features_test.shape[0]):
                idx_test_in_train = np.argmax(np.any(np.all(X_new[:, None] == features_test[i3, idx], axis=2), axis=1))
                features_train_without_test = X_new[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]
                labels_train_without_test = labels_train[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]

                priors = [len(labels_train_without_test[labels_train_without_test == x]) / len(labels_train_without_test) for x in np.unique(labels_train_without_test)]
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                clf.fit(features_train_without_test, labels_train_without_test.flatten())

                X_test = features_test[i3, idx]
                y_test = labels_test[i3]
                if np.max(clf.predict_proba(X_test[np.newaxis, :]))>= 0.5:
                    pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                else:
                    pred_correct_list.append(None)
            pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

            temp_list = []
            for unique_label in np.unique(labels_test):
                correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test==unique_label)] if x is not None])
                percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test==unique_label)])
                temp_list.append(correct_in_class/percent_correct_in_class)
            pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))

        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
    else:
        np.random.seed(42)


        pred_correct_dict_over_n[evaluator_name] = []
        pred_correct_dict_over_n_per_class[evaluator_name] = []
        used_features_idx_over_n[evaluator_name] = {}
        if train_mod.lower() == 'pa':
            features_test = features_test[np.where(labels_test != 'neg control')]
            labels_test = labels_test[np.where(labels_test != 'neg control')]



        X, y = features_train, labels_train

        unique_strings = list(set(y))
        string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
        y = [string_to_int[string] for string in y]

        model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
        results = permutation_importance(model, X, y, scoring='accuracy', n_repeats=1000)
        importance = results.importances_mean


        for no_features in range(1, features_train.shape[1] + 1):
            np.random.seed(42)

            idx = importance >= importance[np.argsort(importance, axis=0)[::-1][no_features - 1]]
            
            X_new = features_train[:, idx]

            used_features_idx_over_n[evaluator_name][no_features] = idx
            pred_correct_list = []

            priors = [len(labels_train[labels_train == x]) / len(labels_train) for x in np.unique(labels_train)]
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
            clf.fit(X_new, labels_train.flatten())
            for i3 in range(features_test.shape[0]):
                X_test = features_test[i3, idx]
                y_test = labels_test[i3]


                if np.max(clf.predict_proba(X_test[np.newaxis, :]))>= 0.5:
                    pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                else:
                    pred_correct_list.append(None)


            pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

            temp_list = []
            for unique_label in np.unique(labels_test):
                correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test==unique_label)] if x is not None])
                percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test==unique_label)])
                temp_list.append(correct_in_class/percent_correct_in_class)
            pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))
        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
def find_optimum_custom(custom_scorer,features_train,labels_train,features_test,labels_test,train_test_identical,train_contains_test,train_mod):
    pred_correct_dict_over_n = {}
    pred_correct_dict_over_n_per_class = {}
    used_features_idx_over_n = {}
    proba_matrix_over_n = {}
    if train_test_identical:
            evaluator_name = str(custom_scorer).split(".")[-1]

            pred_correct_dict_over_n[evaluator_name] = []
            pred_correct_dict_over_n_per_class[evaluator_name] = []
            used_features_idx_over_n[evaluator_name] = {}
            for no_features in range(1, features_train.shape[1] + 1):
                np.random.seed(42)

                try:
                    model = SelectFromModel(custom_scorer, max_features=no_features).fit(features_train, labels_train)
                except:
                    unique_strings = list(set(labels_train))
                    string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
                    labels_train = np.array([string_to_int[string] for string in labels_train])
                    labels_test = np.array([string_to_int[string] for string in labels_test])
                    model = SelectFromModel(custom_scorer, max_features=no_features).fit(features_train, labels_train)

                idx = model.get_support()

                used_features_idx_over_n[evaluator_name][no_features] = idx
                pred_correct_list = []
                X_new = features_train[:, idx]
                for i in range(X_new.shape[0]):
                    X_train = X_new[[x for x in range(X_new.shape[0]) if x != i]]
                    X_test = X_new[i, :]
                    y_train = labels_train[[x for x in range(X_new.shape[0]) if x != i]]
                    y_test = labels_train[i]

                    priors = [len(y_train[y_train == x]) / len(y_train) for x in np.unique(y_train)]

                    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                    clf.fit(X_train, y_train.flatten())


                    if np.max(clf.predict_proba(X_test[np.newaxis, :]))>= 0.5:
                        pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                    else:
                        pred_correct_list.append(None)


                pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

                temp_list = []
                for unique_label in np.unique(labels_test):
                    correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test==unique_label)] if x is not None])
                    percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test==unique_label)])
                    temp_list.append(correct_in_class/percent_correct_in_class)
                pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))
            return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
    elif train_contains_test:

        if train_mod.lower() == 'pa':
            features_test = features_test[np.where(labels_test != 'neg control')]
            labels_test = labels_test[np.where(labels_test != 'neg control')]

        evaluator_name = str(custom_scorer).split(".")[-1]
        pred_correct_dict_over_n[evaluator_name] = []
        pred_correct_dict_over_n_per_class[evaluator_name] = []
        used_features_idx_over_n[evaluator_name] = {}
        for no_features in range(1, features_train.shape[1] + 1):
            np.random.seed(42)

            try:
                model = SelectFromModel(custom_scorer, max_features=no_features).fit(features_train, labels_train)
            except:
                unique_strings = list(set(labels_train))
                string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
                labels_train = np.array([string_to_int[string] for string in labels_train])
                labels_test = np.array([string_to_int[string] for string in labels_test])
                model = SelectFromModel(custom_scorer, max_features=no_features).fit(features_train, labels_train)

            idx = model.get_support()

            used_features_idx_over_n[evaluator_name][no_features] = idx
            pred_correct_list = []

            X_new = features_train[:,idx]
            for i3 in range(features_test.shape[0]):
                idx_test_in_train = np.argmax(np.any(np.all(X_new[:, None] == features_test[i3, idx], axis=2), axis=1))
                features_train_without_test = X_new[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]
                labels_train_without_test = labels_train[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]

                priors = [len(labels_train_without_test[labels_train_without_test == x]) / len(labels_train_without_test) for x in np.unique(labels_train_without_test)]
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                clf.fit(features_train_without_test, labels_train_without_test.flatten())

                X_test = features_test[i3, idx]
                y_test = labels_test[i3]
                if np.max(clf.predict_proba(X_test[np.newaxis, :]))>= 0.5:
                    pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                else:
                    pred_correct_list.append(None)
            pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

            temp_list = []
            for unique_label in np.unique(labels_test):
                correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test==unique_label)] if x is not None])
                percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test==unique_label)])
                temp_list.append(correct_in_class/percent_correct_in_class)
            pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))

        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
    else:
            evaluator_name = str(custom_scorer).split(".")[-1]
            pred_correct_dict_over_n[evaluator_name] = []
            pred_correct_dict_over_n_per_class[evaluator_name] = []
            used_features_idx_over_n[evaluator_name] = {}
            if train_mod.lower() == 'pa':
                features_test = features_test[np.where(labels_test != 'neg control')]
                labels_test = labels_test[np.where(labels_test != 'neg control')]

            for no_features in range(1, features_train.shape[1] + 1):
                np.random.seed(42)

                try:
                    model = SelectFromModel(custom_scorer, max_features=no_features).fit(features_train, labels_train)
                except:
                    unique_strings = list(set(labels_train))
                    string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
                    labels_train = np.array([string_to_int[string] for string in labels_train])
                    labels_test = np.array([string_to_int[string] for string in labels_test])
                    model = SelectFromModel(custom_scorer, max_features=no_features).fit(features_train, labels_train)

                idx = model.get_support()
                X_new = features_train[:, idx]

                used_features_idx_over_n[evaluator_name][no_features] = idx
                pred_correct_list = []

                priors = [len(labels_train[labels_train == x]) / len(labels_train) for x in np.unique(labels_train)]
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                clf.fit(X_new, labels_train.flatten())
                for i3 in range(features_test.shape[0]):
                    X_test = features_test[i3, idx]
                    y_test = labels_test[i3]


                    if np.max(clf.predict_proba(X_test[np.newaxis, :]))>= 0.5:
                        pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                    else:
                        pred_correct_list.append(None)


                pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

                temp_list = []
                for unique_label in np.unique(labels_test):
                    correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test==unique_label)] if x is not None])
                    percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test==unique_label)])
                    temp_list.append(correct_in_class/percent_correct_in_class)
                pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))
            return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
def select_features(features_train: np.ndarray, labels_train: np.ndarray, features_test: np.ndarray, labels_test: np.ndarray, test_mod: str, train_mod: str, plot=False,use_assessment_per_class=False,which_selection='SKB'):
    if features_train.shape == features_test.shape:
        train_test_identical = (features_train == features_test).all()
    else:
        train_test_identical = False
    train_contains_test = np.any(np.all(features_train[:, None] == features_test, axis=2), axis=1).any()
    if which_selection == 'SKB':
        pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_SKB(features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test,train_mod)
    elif which_selection == "PI":
        pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_PI(features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test,train_mod)
    else:
        pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_custom(which_selection,features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test,train_mod)
    if use_assessment_per_class:
        pred_correct_dict_over_n = pred_correct_dict_over_n_per_class

    max_accuracy_idx = np.argmax([np.max(pred_correct_dict_over_n[x]) for x in pred_correct_dict_over_n.keys()])
    if len(np.unique([np.max(pred_correct_dict_over_n[x]) for x in pred_correct_dict_over_n.keys()])) == 1:
        max_accuracy_idx = np.argmin([np.argmax(pred_correct_dict_over_n[x]) for x in pred_correct_dict_over_n.keys()])

    max_accuracy = np.max([np.max(pred_correct_dict_over_n[x]) for x in pred_correct_dict_over_n.keys()])
    max_accuracy_key = list(pred_correct_dict_over_n.keys())[max_accuracy_idx]
    max_accuracy_no_of_feat = np.argmax(pred_correct_dict_over_n[max_accuracy_key]) + 1
    bool_features_2_use = used_features_idx_over_n[max_accuracy_key][max_accuracy_no_of_feat]

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        for evaluator, i in zip(pred_correct_dict_over_n.keys(), range(len(pred_correct_dict_over_n.keys()))):
            ax[i].plot(pred_correct_dict_over_n[evaluator])
            ax[i].title.set_text(f'train: {train_mod}\n'
                                 f'test: {test_mod}\n'
                                 f'Max: {np.max(pred_correct_dict_over_n[evaluator])}\n'
                                 f'Evaluator: {evaluator}')
            ax[i].set_xlabel('no of features')
            ax[i].axvline(np.argmax(pred_correct_dict_over_n[evaluator]), ls='--', lw=2, c='r')
            ax[i].set_xticks(np.arange(0, features_train.shape[1], 3), np.arange(1, features_train.shape[1] + 2, 3))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.1)
        plt.show()

    return bool_features_2_use, max_accuracy_no_of_feat, max_accuracy_key, max_accuracy


if __name__ == "__main__":
    k_means_classes=True
    with_neg_control = True
    np.set_printoptions(suppress=True)

    # Constants
    repeats = 10000
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')



    #load clem cells
    cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['pa','neg_controls','all_cells_new'], load_repaired=True)
    cells.loc[cells['function'].isin(['off-response', 'no response', 'noisy, little modulation']), 'function'] = 'neg_control'
    cells.function = cells.function.apply(lambda x: x.replace(' ',"_"))
    cells = cells.loc[(cells.function != 'nan'), :]
    cells = cells.loc[(~cells.function.isna()), :]
    cells = cells.drop_duplicates(keep='first', inplace=False, subset='cell_name')
    cells = cells.reset_index(drop=True)

    for i,cell in cells.iterrows():
        temp_path = Path(str(cell.metadata_path)[:-4] + "_with_regressor.txt")
        temp_path_pa = path_to_data / 'paGFP'/cell.cell_name/f"{cell.cell_name}_metadata_with_regressor.txt"
        if cell.function == "neg_control":
            cells.loc[i, 'kmeans_function'] = 'neg_control'

        elif temp_path.exists():
            if cell.imaging_modality == 'photoactivation':
                pass
            with open(temp_path,'r') as f:
                t = f.read()
                cells.loc[i, 'kmeans_function'] = t.split('\n')[21].split(' ')[2].strip('"')


        elif temp_path_pa.exists():
            with open(temp_path_pa,'r') as f:
                t = f.read()
                try:
                    cells.loc[i, 'kmeans_function'] = t.split('\n')[17].split(' ')[2].strip('"')
                except:
                    pass

    cells['function'] = cells['kmeans_function']



    #load metrics
    calculate_metric2df(cells, 'FINAL', path_to_data, force_new=False, train_or_predict='train')
    # cells_features = load_train_data_df(path_to_data,'clem_clem_predict_pa_prediction_project_neg_controls')
    # cells_features = cells_features.drop_duplicates(keep='first', inplace=False, subset='cell_name').reset_index(drop=True)
    # cells = cells.loc[(cells.cell_name.isin(cells_features.cell_name)), :]
    # cells = cells.drop_duplicates(keep='first', inplace=False, subset='cell_name').reset_index(drop=True)
    # if not with_neg_control:
    #     cells = cells.loc[~(cells['function'] == 'neg_control'), :]




    # cells_features= cells_features.loc[cells_features['function']!='neg control',:]
    # features = cells_features.iloc[:,5:].to_numpy()
    # labels = cells_features.loc[:,'function'].to_numpy()
    # labels_imaging_modality = cells_features.loc[:,'imaging_modality'].to_numpy()
    # column_labels = cells_features.iloc[:,5:].columns





    file_path = path_to_data / 'make_figures_FK_output' / 'clem_clem_predict_pa_prediction_project_neg_controls_train_features.hdf5'
    all_cells = pd.read_hdf(file_path, 'complete_df')
    all_cells = all_cells.sort_values(by=['function', 'imaging_modality', 'morphology'])

    file_path2 = path_to_data / 'make_figures_FK_output' / 'FINAL_train_features.hdf5'
    all_cells = pd.read_hdf(file_path2, 'complete_df')
    all_cells = all_cells.sort_values(by=['function', 'imaging_modality', 'morphology'])

    # all_cells = all_cells.loc[~all_cells.cell_name.isin(["cell_576460752734566521", "cell_576460752723528109", "cell_576460752684182585"]), :]
    all_cells.loc[all_cells['function'].isin(['no response','off-response','noisy, little modulation']),'function'] = 'neg_control'
    if not with_neg_control:
        all_cells = all_cells.loc[~(all_cells['function']=='neg_control'), :]
    all_cells = all_cells[all_cells['function'] != 'nan']
    all_cells = all_cells.reset_index(drop=True)


    #write kmeans determined function as function
    if k_means_classes:
        for i,cell in all_cells.iterrows():
            kmeans_function = cells.loc[cells['cell_name'] == cell.cell_name, 'kmeans_function'].iloc[0]
            if (kmeans_function!= 'nan') & (cell.function !='neg_control'):
                all_cells.loc[i,'function'] = kmeans_function
        all_cells = all_cells.sort_values(by=['function', 'imaging_modality','morphology'])




    # Impute NaNs
    columns_possible_nans = ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross']
    all_cells.loc[:, columns_possible_nans] = all_cells[columns_possible_nans].fillna(0)

    # Function string replacement
    all_cells.loc[:, 'function'] = all_cells['function'].str.replace('_', ' ')
    all_cells_pa = all_cells[all_cells['imaging_modality'] == 'photoactivation'].copy()
    all_cells_pa.loc[:, 'function'] = all_cells_pa['function'].str.replace('_', ' ')
    all_cells_clem = all_cells[all_cells['imaging_modality'] == 'clem'].copy()
    all_cells_clem.loc[:, 'function'] = all_cells_clem['function'].str.replace('_', ' ')


    # Update 'integrator' function
    def update_integrator(df):
        integrator_mask = df['function'] == 'integrator'
        df.loc[integrator_mask, 'function'] += " " + df.loc[integrator_mask, 'morphology']


    update_integrator(all_cells)
    update_integrator(all_cells_pa)
    update_integrator(all_cells_clem)

    # Replace strings with indices
    columns_replace_string = ['neurotransmitter', 'morphology']
    for work_column in columns_replace_string:
        for i, unique_feature in enumerate(all_cells[work_column].unique()):
            all_cells.loc[all_cells[work_column] == unique_feature, work_column] = i

    # sort by function an imaging modality
    all_cells = all_cells.sort_values(by=['function', 'imaging_modality','morphology'])
    all_cells_pa = all_cells_pa.sort_values(by=['function', 'imaging_modality','morphology'])
    all_cells_clem = all_cells_clem.sort_values(by=['function', 'imaging_modality', 'morphology'])

    # Extract labels
    labels = all_cells['function'].to_numpy()
    labels_imaging_modality = all_cells['imaging_modality'].to_numpy()
    labels_imaging_modality_pa = all_cells_pa['imaging_modality'].to_numpy()
    labels_imaging_modality_clem = all_cells_clem['imaging_modality'].to_numpy()
    labels_pa = all_cells_pa['function'].to_numpy()
    labels_clem = all_cells_clem['function'].to_numpy()
    column_labels = list(all_cells.columns[3:])

    # Extract features
    features = all_cells.iloc[:, 3:].to_numpy()
    features_pa = all_cells[all_cells['imaging_modality'] == 'photoactivation'].iloc[:, 3:].to_numpy()
    features_clem = all_cells[all_cells['imaging_modality'] == 'clem'].iloc[:, 3:].to_numpy()

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features_pa = scaler.transform(features_pa)
    features_clem = scaler.transform(features_clem)


    #REDUCED PLOTS

    # # Train BOTH Test PA
    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features, labels,
    #                                                                                          features_pa, labels_pa,
    #                                                                                          test_mod='PA', train_mod='ALL', plot=True,which_selection=RandomForestClassifier(n_estimators=1000))
    # plot_prediction_matrix_test_train_different(features[:,reduced_features_index],labels,
    #                                             features_pa[:,reduced_features_index],labels_pa,
    #                                             labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
    #                                             title=f'Reduced features\nTrained on Both\nTested on PA',match_limit=0.5)
    #
    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features, labels,
    #                                                                                          features_pa, labels_pa,
    #                                                                                          test_mod='PA', train_mod='ALL', plot=True,use_assessment_per_class=False)
    # plot_prediction_matrix_test_train_different(features[:,reduced_features_index],labels,
    #                                             features_pa[:,reduced_features_index],labels_pa,
    #                                             labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
    #                                             title=f'Reduced features\nTrained on Both\nTested on PA',match_limit=0.5)
    #
    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features, labels,
    #                                                                                          features_pa, labels_pa,
    #                                                                                          test_mod='PA', train_mod='ALL', plot=True,use_assessment_per_class=True)
    # plot_prediction_matrix_test_train_different(features[:,reduced_features_index],labels,
    #                                             features_pa[:,reduced_features_index],labels_pa,
    #                                             labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
    #                                             title=f'Reduced features\nTrained on Both\nTested on PA',match_limit=0.5)
    #



    # Train BOTH Test CLEM

    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features, labels,
                                                                                             features_clem, labels_clem,
                                                                                             test_mod='CLEM', train_mod='ALL', plot=True, which_selection='PI')

    plot_prediction_matrix_test_train_different(features[:,reduced_features_index],labels,
                                                features_clem[:,reduced_features_index],labels_clem,
                                                labels_imaging_modality_clem,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
                                                title=f'Reduced features\nSelected with PI\nTrained on Both\nTested on CLEM',match_limit=0.5)


    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features, labels,
                                                                                             features_clem, labels_clem,
                                                                                             test_mod='CLEM', train_mod='ALL', plot=True,which_selection=RandomForestClassifier(n_estimators=1000))

    plot_prediction_matrix_test_train_different(features[:,reduced_features_index],labels,
                                                features_clem[:,reduced_features_index],labels_clem,
                                                labels_imaging_modality_clem,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
                                                title=f'Reduced features\nSelected with RFC\nTrained on Both\nTested on CLEM',match_limit=0.5)

    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features, labels,
                                                                                             features_clem, labels_clem,
                                                                                             test_mod='CLEM', train_mod='ALL', plot=True, which_selection=DecisionTreeClassifier())

    plot_prediction_matrix_test_train_different(features[:, reduced_features_index], labels,
                                                features_clem[:, reduced_features_index], labels_clem,
                                                labels_imaging_modality_clem, path=path_to_data, column_labels=np.array(column_labels)[reduced_features_index],
                                                title=f'Reduced features\nSelected with CART\nTrained on Both\nTested on CLEM', match_limit=0.5)

    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features, labels,
                                                                                             features_clem, labels_clem,
                                                                                             test_mod='CLEM', train_mod='ALL', plot=True, which_selection=XGBClassifier())

    plot_prediction_matrix_test_train_different(features[:, reduced_features_index], labels,
                                                features_clem[:, reduced_features_index], labels_clem,
                                                labels_imaging_modality_clem, path=path_to_data, column_labels=np.array(column_labels)[reduced_features_index],
                                                title=f'Reduced features\nSelected with XGBClassifier\nTrained on Both\nTested on CLEM', match_limit=0.5)




    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features, labels,
                                                                                             features_clem, labels_clem,
                                                                                             test_mod='CLEM', train_mod='ALL', plot=True)
    plot_prediction_matrix_test_train_different(features[:,reduced_features_index],labels,
                                                features_clem[:,reduced_features_index],labels_clem,
                                                labels_imaging_modality_clem,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
                                                title=f'Reduced features\nSelected with SKB\nTrained on Both\nTested on CLEM',match_limit=0.5)

    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features, labels,
                                                                                             features_clem, labels_clem,
                                                                                             test_mod='CLEM', train_mod='ALL', plot=True,use_assessment_per_class=True)
    plot_prediction_matrix_test_train_different(features[:,reduced_features_index],labels,
                                                features_clem[:,reduced_features_index],labels_clem,
                                                labels_imaging_modality_clem,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
                                                title=f'Reduced features\nSelected with SKB assessment per class\nTrained on Both\nTested on CLEM',match_limit=0.5)

    reduced_features, reduced_features_index, collection_coef_matrix = determine_important_features(features, labels, column_labels, repeats=10000, return_collection_coef_matrix=True, value_automatic_lim=80,
                                                                                                    per_class_selection=None)
    plot_prediction_matrix_test_train_different(features[:,reduced_features_index],labels,
                                                features_clem[:,reduced_features_index],labels_clem,
                                                labels_imaging_modality_clem,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
                                                title=f'Reduced features\nSelected with 10000 repeats\nTrained on Both\nTested on CLEM',match_limit=0.5)



    # # Train BOTH Test BOTH
    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features, labels,
    #                                                                                          features, labels,
    #                                                                                          test_mod='ALL', train_mod='ALL', plot=True)
    # plot_prediction_matrix(features[:,reduced_features_index],labels,
    #                        labels_imaging_modality,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
    #                        title=f'Reduced features\nTrained on Both\nTested on Both',match_limit=0.5)
    #
    # # Train CLEM Test BOTH
    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features_clem, labels_clem,
    #                                                                                          features, labels,
    #                                                                                          test_mod='ALL', train_mod='CLEM', plot=True)
    # plot_prediction_matrix_test_train_different(features_clem[:,reduced_features_index],labels_clem,
    #                                             features[:,reduced_features_index],labels,
    #                                             labels_imaging_modality,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
    #                                             title='Reduced features\nTrained on CLEM\nTested on Both')
    #
    # # Train CLEM Test PA
    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features_clem, labels_clem,
    #                                                                                          features_pa, labels_pa,
    #                                                                                          test_mod='PA', train_mod='CLEM', plot=True)
    # plot_prediction_matrix_test_train_different(features_clem[:,reduced_features_index],labels_clem,
    #                                             features_pa[:,reduced_features_index],labels_pa,
    #                                             labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
    #                                             title='Reduced features\nTrained on CLEM\nTested on PA',train_mod_pa=False)
    #
    # # Train CLEM Test CLEM
    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features_clem, labels_clem,
    #                                                                                          features_clem, labels_clem,
    #                                                                                          test_mod='CLEM', train_mod='CLEM', plot=True)
    # plot_prediction_matrix(features_clem[:,reduced_features_index],labels_clem,
    #                        labels_imaging_modality_clem,path=path_to_data,
    #                        title='Reduced features\nTrained on CLEM\nTested on CLEM',column_labels=np.array(column_labels)[reduced_features_index])
    #
    # # Train PA Test BOTH
    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features_pa, labels_pa,
    #                                                                                          features, labels,
    #                                                                                          test_mod='ALL', train_mod='PA', plot=True)
    # plot_prediction_matrix_test_train_different(features_pa[:,reduced_features_index],labels_pa,
    #                                             features[:,reduced_features_index],labels,labels_imaging_modality,
    #                                             path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
    #                                             title='Reduced features\nTrained on PA\nTested on Both',train_mod_pa=True)
    #
    # # Train PA Test CLEM
    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features_pa, labels_pa,
    #                                                                                          features_clem, labels_clem,
    #                                                                                          test_mod='CLEM', train_mod='PA', plot=True)
    # plot_prediction_matrix_test_train_different(features_pa[:,reduced_features_index],labels_pa,
    #                                             features_clem[:,reduced_features_index],labels_clem,
    #                                             labels_imaging_modality_clem,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
    #                                             title='Reduced features\nTrained on PA\nTested on CLEM',train_mod_pa=True)
    #
    # # Train PA Test PA
    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = select_features(features_pa, labels_pa,
    #                                                                                          features_pa, labels_pa,
    #                                                                                          test_mod='PA', train_mod='PA', plot=True)
    # plot_prediction_matrix(features_pa[:,reduced_features_index],labels_pa,
    #                        labels_imaging_modality_pa,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
    #                        title=f'Reduced features\nTrained on PA\nTested on PA',match_limit=0.5)
    #
    #
    #
    #
    #
    #
    # ##THE PLOT OVER ALL COMBINATIONS FEATURE SEACRCH
    # ##make the plots for feature selection
    # pred_correct_dict = {}
    # for evaluator,evaluator_name in zip([f_classif,mutual_info_classif],['f_classif','mutual_info_classif']):
    #     fig, ax = plt.subplots(3, 3, figsize=(30, 15))
    #     for train_features,train_labels,train_mod,i1  in tqdm(zip([features,features_pa,features_clem],[labels,labels_pa,labels_clem],['both','pa','clem'],range(3)),total=3):
    #         for test_features, test_labels, test_mod,i2 in zip([features, features_pa, features_clem], [labels, labels_pa, labels_clem], ['both', 'pa', 'clem'],range(3)):
    #             if train_mod == "both" and train_mod != test_mod:
    #                 pred_correct_list_over_n = []
    #                 for no_features in range(2, train_features.shape[1] + 1):
    #                     np.random.seed(42)
    #                     SKB = SelectKBest(evaluator, k=no_features)
    #                     X_new = SKB.fit_transform(train_features, train_labels)
    #                     idx = SKB.get_support()
    #                     pred_correct_list = []
    #                     for i3 in range(test_features.shape[0]):
    #                         idx_test_in_train = np.argmax(np.any(np.all(X_new[:, None] == test_features[i3, idx], axis=2), axis=1))
    #                         train_features_without_test = X_new[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]
    #                         train_labels_without_test = train_labels[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]
    #
    #                         priors = [len(train_labels_without_test[train_labels_without_test == x]) / len(train_labels_without_test) for x in np.unique(train_labels_without_test)]
    #                         clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
    #                         clf.fit(train_features_without_test, train_labels_without_test.flatten())
    #
    #
    #                         X_test = test_features[i3, idx]
    #                         y_test = test_labels[i3]
    #
    #                         pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
    #                     pred_correct_dict[f'{train_mod}_{test_mod}'] = pred_correct_list
    #                     pred_correct_list_over_n.append(np.sum(pred_correct_list) / len(pred_correct_list))
    #                 ax[i1][i2].plot(pred_correct_list_over_n)
    #                 ax[i1][i2].title.set_text(f'Train: {train_mod}\nTest: {test_mod}\nMax: {np.max(pred_correct_list_over_n)}')
    #                 ax[i1][i2].set_xlabel('no of features')
    #                 ax[i1][i2].axvline(np.argmax(pred_correct_list_over_n),ls='--',lw=2,c='r')
    #                 ax[i1][i2].set_xticks(np.arange(0, train_features.shape[1], 3), np.arange(2, train_features.shape[1] + 3, 3))
    #             elif train_mod == test_mod:
    #                 pred_correct_list_over_n = []
    #                 for no_features in range(1, train_features.shape[1] + 1):
    #                     np.random.seed(42)
    #                     X_new = SelectKBest(evaluator, k=no_features).fit_transform(train_features, train_labels)
    #                     pred_correct_list = []
    #
    #                     for i in range(X_new.shape[0]):
    #                         X_train = X_new[[x for x in range(X_new.shape[0]) if x != i]]
    #                         X_test = X_new[i, :]
    #                         y_train = train_labels[[x for x in range(X_new.shape[0]) if x != i]]
    #                         y_test = test_labels[i]
    #
    #                         priors = [len(y_train[y_train == x]) / len(y_train) for x in np.unique(y_train)]
    #
    #                         clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
    #                         clf.fit(X_train, y_train.flatten())
    #
    #                         pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
    #                     pred_correct_dict[f'{train_mod}_{test_mod}'] = pred_correct_list
    #                     pred_correct_list_over_n.append(np.sum(pred_correct_list) / len(pred_correct_list))
    #                 ax[i1][i2].plot(pred_correct_list_over_n)
    #                 ax[i1][i2].title.set_text(f'Train: {train_mod}\nTest: {test_mod}\nMax: {np.max(pred_correct_list_over_n)}')
    #                 ax[i1][i2].set_xlabel('no of features')
    #                 ax[i1][i2].axvline(np.argmax(pred_correct_list_over_n),ls='--',lw=2,c='r')
    #                 ax[i1][i2].set_xticks(np.arange(0, train_features.shape[1], 3), np.arange(2, train_features.shape[1] + 3, 3))
    #             elif train_mod != test_mod:
    #                 pred_correct_list_over_n = []
    #                 if train_mod == 'pa':
    #                     test_labels = test_labels[np.where(test_labels!='neg control')]
    #                     test_features = test_features[np.where(test_labels != 'neg control')]
    #
    #                 for no_features in range(2, train_features.shape[1] + 1):
    #                     np.random.seed(42)
    #                     SKB = SelectKBest(evaluator, k=no_features)
    #                     X_new = SKB.fit_transform(train_features, train_labels)
    #                     idx = SKB.get_support()
    #                     pred_correct_list = []
    #                     priors = [len(train_labels[train_labels == x]) / len(train_labels) for x in np.unique(train_labels)]
    #                     clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
    #                     clf.fit(X_new, train_labels.flatten())
    #                     for i3 in range(test_features.shape[0]):
    #                         X_test = test_features[i3, idx]
    #                         y_test = test_labels[i3]
    #
    #                         pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
    #                     pred_correct_dict[f'{train_mod}_{test_mod}'] = pred_correct_list
    #                     pred_correct_list_over_n.append(np.sum(pred_correct_list) / len(pred_correct_list))
    #                 ax[i1][i2].plot(pred_correct_list_over_n)
    #                 ax[i1][i2].title.set_text(f'Train: {train_mod}\nTest: {test_mod}\nMax: {np.max(pred_correct_list_over_n)}')
    #                 ax[i1][i2].set_xlabel('no of features')
    #                 ax[i1][i2].axvline(np.argmax(pred_correct_list_over_n),ls='--',lw=2,c='r')
    #                 ax[i1][i2].set_xticks(np.arange(0, train_features.shape[1], 3), np.arange(2, train_features.shape[1] + 3, 3))
    #
    #     fig.tight_layout()
    #     fig.suptitle(evaluator_name,fontsize='xx-large')
    #
    #     plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.1)
    #     plt.show()
    #
    #
    #
    #
    # select_features(features_pa, labels_pa, features_clem, labels_clem, test_mod='CLEM', train_mod='PA', plot=True)

    # feature selection
    # RANDOM FOREST
    X, y = features[:,reduced_features_index], labels

    unique_strings = list(set(y))
    string_to_int = {string: idx for idx, string in enumerate(y)}

    y = [string_to_int[string] for string in y]


    # define the model
    model = RandomForestClassifier().fit(X, y)
    importance = model.feature_importances_

    #plot
    importance_df = pd.DataFrame(np.stack([np.array(column_labels)[reduced_features_index],importance]).T).sort_values(1).reset_index(drop=True)
    importance_df.iloc[:,1] = importance_df.iloc[:,1].astype(float)
    plt.figure(figsize=(30,15))
    plt.bar(importance_df.index,importance_df[1])
    plt.xticks(np.arange(0,len(importance)),importance_df[0],rotation=45,horizontalalignment='right',verticalalignment='top')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    plt.show()
    plt.figure(figsize=(30, 15))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(np.arange(0, len(importance)), np.array(column_labels)[reduced_features_index], rotation=45, horizontalalignment='right', verticalalignment='top')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    plt.show()


    # CART

    X, y = features, labels

    unique_strings = list(set(y))
    string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
    y = [string_to_int[string] for string in y]

    # define the model
    model = DecisionTreeClassifier().fit(X, y)
    importance = model.feature_importances_

    #plot
    importance_df = pd.DataFrame(np.stack([column_labels,importance]).T).sort_values(1).reset_index(drop=True)
    importance_df.iloc[:,1] = importance_df.iloc[:,1].astype(float)
    plt.figure(figsize=(30,15))
    plt.bar(importance_df.index,importance_df[1])
    plt.xticks(np.arange(0,len(importance)),importance_df[0],rotation=45,horizontalalignment='right',verticalalignment='top')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    plt.show()
    plt.figure(figsize=(30, 15))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(np.arange(0, len(importance)), column_labels, rotation=45, horizontalalignment='right', verticalalignment='top')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    plt.show()


    #XGBOOST
    X, y = features, labels

    unique_strings = list(set(y))
    string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
    y = [string_to_int[string] for string in y]

    model = XGBClassifier().fit(X, y)

    # get importance
    importance = model.feature_importances_
    # summarize feature importance

    # plot feature importance
    importance_df = pd.DataFrame(np.stack([column_labels, importance]).T).sort_values(1).reset_index(drop=True)
    importance_df.iloc[:, 1] = importance_df.iloc[:, 1].astype(float)
    plt.figure(figsize=(30, 15))
    plt.bar(importance_df.index, importance_df[1])
    plt.xticks(np.arange(0, len(importance)), importance_df[0], rotation=45, horizontalalignment='right', verticalalignment='top')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    plt.show()
    plt.figure(figsize=(30, 15))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(np.arange(0, len(importance)), column_labels, rotation=45, horizontalalignment='right', verticalalignment='top')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    plt.show()


    #PERMUTATION FEATURE IMORTANCE

    X, y = features, labels

    unique_strings = list(set(y))
    string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
    y = [string_to_int[string] for string in y]

    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
    results = permutation_importance(model, X, y, scoring='accuracy',n_repeats=1000)
    importance = results.importances_mean

    #plot
    importance_df = pd.DataFrame(np.stack([column_labels,importance]).T)
    importance_df.iloc[:,1] = importance_df.iloc[:,1].astype(np.float64)
    importance_df = importance_df.sort_values(1).reset_index(drop=True)
    plt.figure(figsize=(30,15))
    plt.bar(importance_df.index,importance_df[1])
    plt.xticks(np.arange(0,len(importance)),importance_df[0],rotation=45,horizontalalignment='right',verticalalignment='top')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    plt.show()
    plt.figure(figsize=(30, 15))
    plt.bar([x for x in range(len(importance))], importance)
    plt.xticks(np.arange(0, len(importance)), column_labels, rotation=45, horizontalalignment='right', verticalalignment='top')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    plt.show()

