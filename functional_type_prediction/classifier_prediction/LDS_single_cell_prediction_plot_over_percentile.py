from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.patches import Patch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.calculate_metric2df import *



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
def plot_prediction_matrix_test_train_different(features_train,labels_train,features_test,labels_test,labels_imaging_modality,path,column_labels,solver='lsqr',shrinkage='auto',title='prediction_plot',match_limit=0.5,return_metrics=False):
    #init variables
    features = features_test
    labels = labels_test
    prob_matrix = np.empty(shape=(features.shape[0],len(np.unique(labels_train))))
    pred_matrix = np.empty(shape=(features.shape[0],1),dtype='<U24')
    pred_correct = np.empty(shape=(features.shape[0], 1))
    no_predict = np.empty(shape=(features.shape[0], 1))

    used_features= []
    for label in column_labels:
        used_features.append(Patch(facecolor='white', edgecolor='white', label=label))
    priors = [len(labels[labels == x]) / len(labels) for x in np.unique(labels_train)]
    #predict all cells one by one
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, priors=priors)
    clf.fit(features_train, labels_train.flatten())

    for i in range(features.shape[0]):
        X_train = features[[x for x in range(features.shape[0]) if x != i]]
        X_test = features[i,:]
        y_train = labels[[x for x in range(features.shape[0]) if x != i]]
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
def determine_important_features(features,labels,feature_labels, repeats=10000,random_seed=42,solver='lsqr',shrinkage='auto',test_size=0.3,stratify=True,return_collection_coef_matrix=False,value_automatic_lim=80):
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

    if value_automatic_lim is not None:
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
    k_means_classes=False
    with_neg_control = True

    np.set_printoptions(suppress=True)

    # Constants
    repeats = 10000
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')



    #load clem cells
    cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem','clem_predict', 'pa','prediction_project','neg_controls','all_cells_new'], load_repaired=True)
    cells.loc[cells['function'].isin(['off-response', 'no response', 'noisy, little modulation']), 'function'] = 'neg_control'
    cells = cells.loc[(cells.function != 'nan'), :]
    cells = cells.loc[(~cells.function.isna()), :]
    cells = cells.drop_duplicates(keep='first', inplace=False, subset='cell_name')
    cells = cells.reset_index(drop=True)

    #load metrics
    calculate_metric2df(cells, 'clem_clem_predict_pa_prediction_project_neg_controls', path_to_data, force_new=False, train_or_predict='train')
    cells_features = load_train_data_df(path_to_data,'clem_clem_predict_pa_prediction_project_neg_controls')
    cells_features = cells_features.drop_duplicates(keep='first', inplace=False, subset='cell_name').reset_index(drop=True)
    cells = cells.loc[(cells.cell_name.isin(cells_features.cell_name)), :]
    cells = cells.drop_duplicates(keep='first', inplace=False, subset='cell_name').reset_index(drop=True)
    if not with_neg_control:
        cells = cells.loc[~(cells['function'] == 'neg_control'), :]

    #adjust based on regressor
    cells['prediction_equals_manual'] = False
    cells['correlation_test_passed'] = False
    cells['prediction_regressor'] = np.nan
    cells['prediction_equals_manual_st'] = False
    cells['correlation_test_passed_st'] = False
    cells['prediction_regressor_st'] = np.nan

    # for i,cell in cells.iterrows():
    #     temp_path = Path(str(cell.metadata_path)[:-4] + "_with_regressor.txt")
    #     temp_path_pa = path_to_data / 'paGFP'/cell.cell_name/f"{cell.cell_name}_metadata_with_regressor.txt"
    #     if temp_path.exists():
    #         if cell.imaging_modality == 'photoactivation':
    #             pass
    #         with open(temp_path,'r') as f:
    #             t = f.read()
    #
    #             cells.loc[i,'prediction_regressor'] = t.split('\n')[15].split(' ')[2].strip('"')
    #             cells.loc[i,'correlation_test_passed'] = eval(t.split('\n')[16].split(' ')[2].strip('"'))
    #             cells.loc[i, 'prediction_equals_manual'] = eval(t.split('\n')[17].split(' ')[2].strip('"'))
    #
    #             cells.loc[i,'prediction_regressor_st'] = t.split('\n')[18].split(' ')[2].strip('"')
    #             cells.loc[i,'correlation_test_passed_st'] = eval(t.split('\n')[19].split(' ')[2].strip('"'))
    #             cells.loc[i, 'prediction_equals_manual_st'] = eval(t.split('\n')[20].split(' ')[2].strip('"'))
    #             cells.loc[i, 'kmeans_function'] = t.split('\n')[21].split(' ')[2].strip('"')
    #
    #
    #     elif temp_path_pa.exists():
    #         with open(temp_path_pa,'r') as f:
    #             t = f.read()
    #             cells.loc[i,'prediction_regressor'] = t.split('\n')[11].split(' ')[2].strip('"')
    #             cells.loc[i,'correlation_test_passed'] = eval(t.split('\n')[12].split(' ')[2].strip('"'))
    #             cells.loc[i, 'prediction_equals_manual'] = eval(t.split('\n')[13].split(' ')[2].strip('"'))
    #
    #             cells.loc[i,'prediction_regressor_st'] = t.split('\n')[14].split(' ')[2].strip('"')
    #             cells.loc[i,'correlation_test_passed_st'] = eval(t.split('\n')[15].split(' ')[2].strip('"'))
    #             cells.loc[i, 'prediction_equals_manual_st'] = eval(t.split('\n')[16].split(' ')[2].strip('"'))
    #             try:
    #                 cells.loc[i, 'kmeans_function'] = t.split('\n')[17].split(' ')[2].strip('"')
    #             except:
    #                 pass





    cells_features= cells_features.loc[cells_features['function']!='neg control',:]
    features = cells_features.iloc[:,5:].to_numpy()
    labels = cells_features.loc[:,'function'].to_numpy()
    labels_imaging_modality = cells_features.loc[:,'imaging_modality'].to_numpy()
    column_labels = cells_features.iloc[:,5:].columns





    file_path = path_to_data / 'make_figures_FK_output' / 'clem_clem_predict_pa_prediction_project_neg_controls_train_features.hdf5'
    all_cells = pd.read_hdf(file_path, 'complete_df')


    all_cells = all_cells.loc[~all_cells.cell_name.isin(["cell_576460752734566521", "cell_576460752723528109", "cell_576460752684182585"]), :]
    all_cells.loc[all_cells['function'].isin(['no response','off-response']),'function'] = 'neg_control'
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

    # LDA

    reduced_features,reduced_features_index,collection_coef_matrix = determine_important_features(features,labels,column_labels,repeats=10000,return_collection_coef_matrix=True)
    reduced_features2, reduced_features_index2 = determine_important_features_RFECV(features, labels, column_labels, scoring='roc_auc_ovo')


    both_both_list = []
    bb_dt = []
    bb_ic = []
    bb_ii = []
    bb_mc = []
    bb_nc = []

    pa_clem_list = []
    pc_dt = []
    pc_ic = []
    pc_ii = []
    pc_mc = []
    pc_nc = []

    pa_pa_list = []
    pp_dt = []
    pp_ic = []
    pp_ii = []
    pp_mc = []
    pp_nc = []

    
    for i in np.arange(0,100,1):
        print(i)
        reduced_features, reduced_features_index, collection_coef_matrix = determine_important_features(features, labels, column_labels, repeats=5000, return_collection_coef_matrix=True,value_automatic_lim=i)
        n_predictions_correct, n_predictions_incorrect, n_no_predicitons,n_correct_per_class_dict,percent_correct_per_class_dict =     plot_prediction_matrix(features[:,reduced_features_index],labels,labels_imaging_modality,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],title=f'Reduced features\nTrained on Both\nTested on Both',match_limit=0.5,return_metrics=True)

        both_both_list.append((n_predictions_correct/features.shape[0])*100)
        bb_dt.append(percent_correct_per_class_dict["dynamic threshold"]*100)
        bb_ic.append(percent_correct_per_class_dict["integrator contralateral"]*100)
        bb_ii.append(percent_correct_per_class_dict["integrator ipsilateral"]*100)
        bb_mc.append(percent_correct_per_class_dict["motor command"]*100)
        bb_nc.append(percent_correct_per_class_dict["neg control"]*100)
        
        n_predictions_correct, n_predictions_incorrect, n_no_predicitons,n_correct_per_class_dict,percent_correct_per_class_dict =     plot_prediction_matrix_test_train_different(features_pa[:,reduced_features_index],labels_pa,features_clem[:,reduced_features_index],labels_clem,labels_imaging_modality_clem,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],
                           title='Reduced features\nTrained on PA\nTested on CLEM',return_metrics=True)
        pa_clem_list.append((n_predictions_correct/features_clem.shape[0])*100)
        pc_dt.append(percent_correct_per_class_dict["dynamic threshold"]*100)
        pc_ic.append(percent_correct_per_class_dict["integrator contralateral"]*100)
        pc_ii.append(percent_correct_per_class_dict["integrator ipsilateral"]*100)
        pc_mc.append(percent_correct_per_class_dict["motor command"]*100)
        pc_nc.append(percent_correct_per_class_dict["neg control"]*100)
        
        n_predictions_correct, n_predictions_incorrect, n_no_predicitons,n_correct_per_class_dict,percent_correct_per_class_dict =     plot_prediction_matrix(features_pa[:,reduced_features_index],labels_pa,labels_imaging_modality,path=path_to_data,column_labels=np.array(column_labels)[reduced_features_index],title=f'Reduced features\nTrained on Both\nTested on Both',match_limit=0.5,return_metrics=True)

        
        pa_pa_list.append((n_predictions_correct/features_clem.shape[0])*100)
        pp_dt.append(percent_correct_per_class_dict["dynamic threshold"]*100)
        pp_ic.append(percent_correct_per_class_dict["integrator contralateral"]*100)
        pp_ii.append(percent_correct_per_class_dict["integrator ipsilateral"]*100)
        pp_mc.append(percent_correct_per_class_dict["motor command"]*100)

    plt.figure(figsize=(20, 10))
    plt.plot(both_both_list,label='bb')
    plt.plot(np.array(bb_dt)/100, label='bb_dt')
    plt.plot(np.array(bb_ic)/100, label='bb_ic')
    plt.plot(np.array(bb_ii)/100, label='bb_ii')
    plt.plot(np.array(bb_mc)/100, label='bb_mc')
    plt.plot(np.array(bb_nc)/100, label='bb_nc')
    plt.xticks(np.arange(0, 100, 10), np.arange(0, 100, 10))
    plt.legend(bbox_to_anchor=(1.1, 0.975))
    plt.subplots_adjust(left=0.1, right=0.8, top=0.7, bottom=0.1)
    plt.xlabel('quantile for selection')
    plt.ylabel('% correct')
    plt.title('prediction over different selection cutoffs both train both test')
    plt.show()

    plt.figure(figsize=(20,10))
    plt.plot(pa_clem_list, label='pa_clem')
    plt.plot(np.array(pc_dt) / 100, label='pc_dt')
    plt.plot(np.array(pc_ic) / 100, label='pc_ic')
    plt.plot(np.array(pc_ii) / 100, label='pc_ii')
    plt.plot(np.array(pc_mc) / 100, label='pc_mc')
    plt.plot(np.array(pc_nc) / 100, label='pc_nc')
    plt.xticks(np.arange(0, 100, 10), np.arange(0, 100, 10))
    plt.legend(bbox_to_anchor=(1.1, 0.975))
    plt.subplots_adjust(left=0.1, right=0.8, top=0.7, bottom=0.1)
    plt.xlabel('quantile for selection')
    plt.ylabel('% correct')
    plt.title('prediction over different selection cutoffs pa train clem test')
    plt.show()
    
    plt.figure(figsize=(20,10))
    plt.plot(pa_clem_list, label='pa_clem')
    plt.plot(np.array(pp_dt) / 100, label='pp_dt')
    plt.plot(np.array(pp_ic) / 100, label='pp_ic')
    plt.plot(np.array(pp_ii) / 100, label='pp_ii')
    plt.plot(np.array(pp_mc) / 100, label='pp_mc')
    plt.xticks(np.arange(0, 100, 10), np.arange(0, 100, 10))
    plt.legend(bbox_to_anchor=(1.1, 0.975))
    plt.subplots_adjust(left=0.1, right=0.8, top=0.7, bottom=0.1)
    plt.xlabel('quantile for selection')
    plt.ylabel('% correct')
    plt.title('prediction over different selection cutoffs pa train pa test')
    plt.show()

