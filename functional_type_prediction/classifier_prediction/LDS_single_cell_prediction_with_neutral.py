import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.patches import Patch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.calculate_metric2df import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.calculate_metric2df_old import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.calculate_metric2df import *
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import LeavePOut
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
import matplotlib.patheffects as PathEffects
from scipy.stats import entropy
from scipy.stats import hmean
import sklearn
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
def calc_penalty(temp_list):
    p = 2  # You can adjust this power as needed
    penalty = np.mean(temp_list) / np.exp(np.std(temp_list) ** p)
    return penalty
def do_n_repeat_train_test_confusion_matrix_lpo(features,labels,test_name,p=5):
    plt.figure(figsize=(10,10))
    # Initialize variables to store true and predicted labels
    true_labels = []
    pred_labels = []

    lpo = LeavePOut(p=p)

    # Run train_test_split 100 times
    for train, test in lpo.split(features,labels):
        # Split the data
        try:
            X_train, X_test, y_train, y_test = features[train,:],features[test,:],labels[train],labels[test]

            # Train the model
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors).fit(X_train, y_train)


            # Predict on the test set
            y_pred = clf.predict(X_test)

            # Store true and predicted labels
            true_labels.extend(y_test)
            pred_labels.extend(y_pred)
        except:
            pass

    # Convert lists to numpy arrays (optional, but can be useful)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels,normalize='true').astype(float)


    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
    plt.title(f"Confusion Matrix (LPO = {p})"+f'\n{test_name}')
    plt.show()
def do_n_repeat_train_test_confusion_matrix_kfold(features,labels,test_name,test_size=0.3,n_repeats=100,n_splits=5):
        plt.figure(figsize=(10,10))
        # Initialize variables to store true and predicted labels
        true_labels = []
        pred_labels = []

        rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

        # Run train_test_split 100 times
        for train, test in rkf.split(features,labels):
            # Split the data
            X_train, X_test, y_train, y_test = features[train,:],features[test,:],labels[train],labels[test]

            # Train the model
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors).fit(X_train, y_train)


            # Predict on the test set
            y_pred = clf.predict(X_test)

            # Store true and predicted labels
            true_labels.extend(y_test)
            pred_labels.extend(y_pred)

        # Convert lists to numpy arrays (optional, but can be useful)
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        # Create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels,normalize='true').astype(float)


        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
        plt.title(f"Confusion Matrix (after {n_repeats} KFold {n_splits} splits)"+f'\n{test_name}')
        plt.show()
def do_n_repeat_train_test_confusion_matrix_simple_split(features,labels,test_name,test_size=0.3,n=100,ax=None):
        plt.figure(figsize=(10,10))
        # Initialize variables to store true and predicted labels
        true_labels = []
        pred_labels = []

        # Run train_test_split 100 times
        for i in range(n):
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=i)

            # Train the model
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors).fit(X_train, y_train)


            # Predict on the test set
            y_pred = clf.predict(X_test)

            # Store true and predicted labels
            true_labels.extend(y_test)
            pred_labels.extend(y_pred)

        # Convert lists to numpy arrays (optional, but can be useful)
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        # Create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels,normalize='true').astype(float)

        # Plot confusion matrix
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
            ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
            plt.title(f"Confusion Matrix (after {n} 70:30 splits)"+f'\n{test_name}')
            plt.show()
        else:
            ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
            ax.title.set_text(f"Confusion Matrix (after {n} 70:30 splits)"+f'\n{test_name}')


def do_n_repeat_train_test_confusion_matrix_simple_split_test_not_train(features_train, labels_train,
                                                                        features_test, labels_test, test_name, test_size=0.3, n=100,ax=None):
    plt.figure(figsize=(10, 10))
    # Initialize variables to store true and predicted labels
    true_labels = []
    pred_labels = []

    # Run train_test_split 100 times
    for i in range(n):
        # Split the data
        X_train, _, y_train, _ = train_test_split(features_train, labels_train, test_size=test_size, random_state=i)
        _, X_test, _, y_test = train_test_split(features_test, labels_test, test_size=test_size, random_state=i)

        # Train the model
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors).fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Store true and predicted labels
        true_labels.extend(y_test)
        pred_labels.extend(y_pred)

    # Convert lists to numpy arrays (optional, but can be useful)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, normalize='true').astype(float)

    # Plot confusion matrix
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
        plt.title(f"Confusion Matrix (after {n} 70:30 splits)" + f'\n{test_name}')
        plt.show()
    else:
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
        ax.title.set_text(f"Confusion Matrix (after {n} 70:30 splits)"+f'\n{test_name}')
def do_n_repeat_train_test_confusion_matrix_simple_split_test_not_train_train_contains_test(features_train, labels_train,
                                                                        features_test, labels_test, test_name, test_size=0.3, n=100,ax=None):
    plt.figure(figsize=(10, 10))
    # Initialize variables to store true and predicted labels
    true_labels = []
    pred_labels = []

    # Run train_test_split 100 times
    for i in range(n):
        # Split the data
        X_train1, _, y_train1, _ = train_test_split(features_train, labels_train, test_size=test_size, random_state=i)
        X_train2, X_test, y_train2, y_test = train_test_split(features_test, labels_test, test_size=test_size, random_state=i)

        X_train, y_train = np.concatenate([X_train1,X_train2]), np.concatenate([y_train1,y_train2])
        # Train the model
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors).fit(X_train, y_train)

        # Predict on the test set
        y_pred = clf.predict(X_test)

        # Store true and predicted labels
        true_labels.extend(y_test)
        pred_labels.extend(y_pred)

    # Convert lists to numpy arrays (optional, but can be useful)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, normalize='true').astype(float)

    # Plot confusion matrix
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
        plt.title(f"Confusion Matrix (after {n} 70:30 splits)" + f'\n{test_name}')
        plt.show()
    else:
        ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
        ax.title.set_text(f"Confusion Matrix (after {n} 70:30 splits)"+f'\n{test_name}')
def plot_prediction_matrix(features,labels,labels_imaging_modality,path,column_labels,solver='lsqr',shrinkage='auto',title='prediction_plot',match_limit=0.5,return_metrics=False,ax=None):
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
    prediction_matrix_dict = {}

    for unique_label in np.unique(labels):
        n_correct_per_class_dict[unique_label] = np.sum(pred_matrix[np.where(labels == unique_label)] == unique_label)
        percent_correct_per_class_dict[unique_label] = convert_percent_specific(n_correct_per_class_dict[unique_label], np.sum(labels == unique_label))
        prediction_matrix_dict[unique_label] = {}
        for unique_label2 in np.unique(labels):
            prediction_matrix_dict[unique_label][unique_label2] = np.sum(pred_matrix[np.where(labels == unique_label)] == unique_label2)
        prediction_matrix_dict[unique_label]['n_no_prediction'] = np.sum(no_predict[np.where(labels == unique_label)])
        prediction_matrix_dict[unique_label]['n_not_correct'] = np.sum(pred_matrix[np.where(labels == unique_label)] != unique_label)
        prediction_matrix_dict[unique_label]['n_correct'] = np.sum(pred_matrix[np.where(labels == unique_label)] == unique_label)
        prediction_matrix_dict[unique_label]['n_total'] = pred_matrix[np.where(labels == unique_label)].shape[0]


    if return_metrics:



        return n_predictions_correct, n_predictions_incorrect, n_no_predicitons,n_correct_per_class_dict,percent_correct_per_class_dict
    else:
        if ax is None:
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
            plt.savefig(savepath /'pdf'/ (title.replace('\n'," ").replace("/","_") + ".pdf"))
            plt.savefig(savepath /'png'/ (title.replace('\n'," ").replace("/","_")+ ".png"))

            path_to_open = savepath /'pdf'/ (title.replace('\n'," ").replace("/","_")+ ".pdf")

            os.startfile(path_to_open)

            plt.show()
        else:
            df = pd.DataFrame.from_dict(prediction_matrix_dict, orient='index').astype(int)
            df_array = df.iloc[:, :5].to_numpy()
            df_normed = (df.iloc[:, :5] / df.iloc[:, -1]).fillna(0)
            df_normed_array = df_normed.to_numpy()

            # fig, ax = plt.subplots(1, 1,figsize=(15,15))
            ax.pcolormesh(df_normed.T, vmax=1, vmin=0, )
            ax.set_xticks(np.arange(0, 4) + 0.5, [f'{x}\n({df.loc[x, "n_total"]} cells)' for x in df_normed.index], ha='center', va='bottom')
            ax.set_yticks(np.arange(0, 5) + 0.5, df_normed.columns, rotation=0, ha='left', va='center')
            ax.xaxis.tick_top()

            ax.set_title(title + f'\nPredictions correct: {convert_percent(n_predictions_correct)}%'
                                 f'\nPredictions incorrect: {convert_percent(n_predictions_incorrect)}%'
                                 f'\nNo prediction: {convert_percent(n_no_predicitons)}%\n')

            for i1 in range(df_array.T.shape[0]):
                for i2 in range(df_array.T.shape[1]):
                    text = ax.text(i2 + 0.5, i1 + 0.5, f'{df_array[i2, i1]}', ha='center',
                                   va='center', font={'weight': 'heavy'}, fontsize=50, c='white')
                    text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
            ax.invert_yaxis()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', length=0)
            ax.tick_params(axis='y', which='both', length=0)
            ax.axis('equal')
            ax.set_xlim(0, np.min(df_array.shape))
            plt.tight_layout()
            return convert_percent(n_predictions_correct)
def plot_prediction_matrix_test_train_different(features_train,labels_train,features_test,labels_test,labels_imaging_modality,path,column_labels,train_mod_pa=False,solver='lsqr',shrinkage='auto',title='prediction_plot',match_limit=0.5,return_metrics=False,ax=None):
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
    prediction_matrix_dict = {}

    for unique_label in np.unique(labels):
        n_correct_per_class_dict[unique_label] = np.sum(pred_matrix[np.where(labels == unique_label)] == unique_label)
        percent_correct_per_class_dict[unique_label] = convert_percent_specific(n_correct_per_class_dict[unique_label], np.sum(labels == unique_label))
        prediction_matrix_dict[unique_label] = {}
        for unique_label2 in np.unique(labels):
            prediction_matrix_dict[unique_label][unique_label2] =  np.sum(pred_matrix[np.where(labels == unique_label)] == unique_label2)
        prediction_matrix_dict[unique_label]['n_no_prediction'] = np.sum(no_predict[np.where(labels == unique_label)])
        prediction_matrix_dict[unique_label]['n_not_correct'] = np.sum(pred_matrix[np.where(labels == unique_label)] != unique_label)
        prediction_matrix_dict[unique_label]['n_correct'] = np.sum(pred_matrix[np.where(labels == unique_label)] == unique_label)
        prediction_matrix_dict[unique_label]['n_total'] = pred_matrix[np.where(labels == unique_label)].shape[0]





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


        if ax is None:
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
            plt.savefig(savepath /'pdf'/ (title.replace('\n'," ").replace("/","_") + ".pdf"))
            plt.savefig(savepath /'png'/ (title.replace('\n'," ").replace("/","_")+ ".png"))

            path_to_open = savepath /'pdf'/ (title.replace('\n'," ").replace("/","_")+ ".pdf")

            os.startfile(path_to_open)

            plt.show()
        else:


            df = pd.DataFrame.from_dict(prediction_matrix_dict, orient='index').astype(int)
            df_array = df.iloc[:, :5].to_numpy()
            df_normed = (df.iloc[:, :5] / df.iloc[:, -1]).fillna(0)
            df_normed_array = df_normed.to_numpy()

            # fig, ax = plt.subplots(1, 1,figsize=(15,15))
            ax.pcolormesh(df_normed.T, vmax=1, vmin=0, )
            ax.set_xticks(np.arange(0, 4) + 0.5, [f'{x}\n({df.loc[x, "n_total"]} cells)' for x in df_normed.index], ha='center', va='bottom')
            ax.set_yticks(np.arange(0, 5) + 0.5, df_normed.columns, rotation=0, ha='left', va='center')
            ax.xaxis.tick_top()


            ax.set_title(title + f'\nPredictions correct: {convert_percent(n_predictions_correct)}%'
                              f'\nPredictions incorrect: {convert_percent(n_predictions_incorrect)}%'
                              f'\nNo prediction: {convert_percent(n_no_predicitons)}%\n')

            for i1 in range(df_array.T.shape[0]):
                for i2 in range(df_array.T.shape[1]):
                    text = ax.text(i2+0.5, i1+0.5, f'{df_array[i2,i1]}', ha='center',
                            va='center', font={'weight': 'heavy' },fontsize=50,c='white')
                    text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
            ax.invert_yaxis()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(axis='x', which='both', length=0)
            ax.tick_params(axis='y', which='both', length=0)
            ax.axis('equal')
            ax.set_xlim(0, np.min(df_array.shape))
            plt.tight_layout()
            return convert_percent(n_predictions_correct)
            # plt.show()
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
def find_optimum_SKB(features_train,labels_train,features_test,labels_test,train_test_identical,train_contains_test,train_mod,use_std_scale=False):
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
                if use_std_scale:
                    penalty = calc_penalty(temp_list)
                    pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
                else:
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
                if use_std_scale:
                    penalty = calc_penalty(temp_list)
                    pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
                else:
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
                if use_std_scale:
                    penalty = calc_penalty(temp_list)
                    pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
                else:
                    pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))

        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
def find_optimum_PI(features_train,labels_train,features_test,labels_test,train_test_identical,train_contains_test,train_mod,use_std_scale=False):
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
            if use_std_scale:
                penalty = calc_penalty(temp_list)
                pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
            else:
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
            if use_std_scale:
                penalty = calc_penalty(temp_list)
                pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
            else:
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
            if use_std_scale:
                penalty = calc_penalty(temp_list)
                pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
            else:
                pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))

        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
def find_optimum_custom(custom_scorer,features_train,labels_train,features_test,labels_test,train_test_identical,train_contains_test,train_mod,use_std_scale=False):
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
                if use_std_scale:
                    penalty = calc_penalty(temp_list)
                    pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
                else:
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
            if use_std_scale:
                penalty = calc_penalty(temp_list)
                pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
            else:
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
                if use_std_scale:
                    penalty = calc_penalty(temp_list)
                    pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
                else:
                    pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))

            return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
def select_features(features_train: np.ndarray, labels_train: np.ndarray, features_test: np.ndarray, labels_test: np.ndarray, test_mod: str, train_mod: str, plot=False,use_assessment_per_class=False,which_selection='SKB',use_std_scale=False):
    if features_train.shape == features_test.shape:
        train_test_identical = (features_train == features_test).all()
    else:
        train_test_identical = False
    train_contains_test = np.any(np.all(features_train[:, None] == features_test, axis=2), axis=1).any()
    if which_selection == 'SKB':
        pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_SKB(features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test,train_mod,use_std_scale=use_std_scale)
    elif which_selection == "PI":
        pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_PI(features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test,train_mod,use_std_scale=use_std_scale)
    else:
        pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_custom(which_selection,features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test,train_mod,use_std_scale=use_std_scale)
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

    return bool_features_2_use, max_accuracy_no_of_feat, max_accuracy_key, max_accuracy,train_mod,test_mod
def prepare_data_4_metric_calc(df, use_new_neurotransmitter, use_k_means_classes,path_to_data,train_or_predict='train'):
    if train_or_predict == 'train':
        df.loc[df['function'].isin(['off-response', 'no response', 'noisy, little modulation']), 'function'] = 'neg_control'
        df.function = df.function.apply(lambda x: x.replace(' ', "_"))
        df = df.loc[(df.function != 'nan'), :]
        df = df.loc[(~df.function.isna()), :]
    df = df.drop_duplicates(keep='first', inplace=False, subset='cell_name')
    df = df.reset_index(drop=True)
    if train_or_predict == 'train':
        if use_k_means_classes:
            for i, cell in df.iterrows():
                temp_path = Path(str(cell.metadata_path)[:-4] + "_with_regressor.txt")
                temp_path_pa = path_to_data / 'paGFP' / cell.cell_name / f"{cell.cell_name}_metadata_with_regressor.txt"
                if cell.function == "neg_control":
                    df.loc[i, 'kmeans_function'] = 'neg_control'

                elif temp_path.exists():
                    if cell.imaging_modality == 'photoactivation':
                        pass
                    with open(temp_path, 'r') as f:
                        t = f.read()
                        df.loc[i, 'kmeans_function'] = t.split('\n')[21].split(' ')[2].strip('"')


                elif temp_path_pa.exists():
                    with open(temp_path_pa, 'r') as f:
                        t = f.read()
                        try:
                            df.loc[i, 'kmeans_function'] = t.split('\n')[17].split(' ')[2].strip('"')
                        except:
                            pass

            df['function'] = df['kmeans_function']
        if use_new_neurotransmitter:
            new_neurotransmitter = pd.read_excel(path_to_data / 'em_zfish1' / 'Figures' / 'Fig 4' / 'cells2show.xlsx', sheet_name='paGFP stack quality', dtype=str)

            neurotransmitter_dict = {'Vglut2a': 'excitatory', 'Gad1b': 'inhibitory'}
            for i, cell in df.iterrows():
                if cell.imaging_modality == 'photoactivation':
                    if new_neurotransmitter.loc[new_neurotransmitter['Name'] == cell.cell_name, 'Neurotransmitter'].iloc[0] is np.nan:
                        df.loc[i, 'neurotransmitter'] == 'nan'
                    else:
                        df.loc[i, 'neurotransmitter'] = neurotransmitter_dict[new_neurotransmitter.loc[new_neurotransmitter['Name'] == cell.cell_name, 'Neurotransmitter'].iloc[0]]

    return df

def load_metrics_train(file_name,path_to_data,with_neg_control=False):

    file_path = path_to_data / 'make_figures_FK_output' / f'{file_name}_train_features.hdf5'

    all_cells = pd.read_hdf(file_path, 'complete_df')

    # throw out weird jon cells
    # all_cells = all_cells.loc[~all_cells.cell_name.isin(["cell_576460752734566521", "cell_576460752723528109", "cell_576460752684182585"]), :]

    #Data Preprocessing
    all_cells = all_cells[all_cells['function'] != 'nan']
    all_cells = all_cells.sort_values(by=['function', 'morphology', 'imaging_modality', 'neurotransmitter'])
    all_cells = all_cells.reset_index(drop=True)


    all_cells.loc[all_cells['function'].isin(['no response', 'off-response', 'noisy, little modulation']), 'function'] = 'neg_control'
    if not with_neg_control:
        all_cells = all_cells.loc[~(all_cells['function'] == 'neg_control'), :]

    # Impute NaNs
    columns_possible_nans = ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross']
    all_cells.loc[:, columns_possible_nans] = all_cells[columns_possible_nans].fillna(0)

    # Function string replacement
    all_cells.loc[:, 'function'] = all_cells['function'].str.replace('_', ' ')

    # Update 'integrator' function
    def update_integrator(df):
        integrator_mask = df['function'] == 'integrator'
        df.loc[integrator_mask, 'function'] += " " + df.loc[integrator_mask, 'morphology']

    update_integrator(all_cells)


    # Replace strings with indices

    columns_replace_string = ['neurotransmitter', 'morphology']
    neurotransmitter2int_dict = {'excitatory': 0, 'inhibitory': 1, 'na': 2, 'nan': 2}
    morphology2int_dict = {'contralateral': 0, 'ipsilateral': 1}

    for work_column in columns_replace_string:
        all_cells.loc[:, work_column + "_clone"] = all_cells[work_column]
        for key in eval(f'{work_column}2int_dict').keys():
            all_cells.loc[all_cells[work_column] == key, work_column] = eval(f'{work_column}2int_dict')[key]



    # Extract labels
    labels = all_cells['function'].to_numpy()
    labels_imaging_modality = all_cells['imaging_modality'].to_numpy()
    column_labels = list(all_cells.columns[3:-len(columns_replace_string)])

    # Extract features
    features = all_cells.iloc[:, 3:-len(columns_replace_string)].to_numpy()

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return [features,labels,labels_imaging_modality],column_labels,all_cells

def load_metrics_predict(file_name,path_to_data,with_neg_control=False):
    file_path = path_to_data / 'make_figures_FK_output' / f'{file_name}_predict_features.hdf5'

    all_cells = pd.read_hdf(file_path, 'complete_df')

    # throw out weird jon cells
    # all_cells = all_cells.loc[~all_cells.cell_name.isin(["cell_576460752734566521", "cell_576460752723528109", "cell_576460752684182585"]), :]

    # Data Preprocessing
    all_cells = all_cells.sort_values(by=['morphology','neurotransmitter'])
    all_cells = all_cells.reset_index(drop=True)

    # Impute NaNs
    columns_possible_nans = ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross']
    all_cells.loc[:, columns_possible_nans] = all_cells[columns_possible_nans].fillna(0)


    # Replace strings with indices

    columns_replace_string = ['neurotransmitter', 'morphology']
    neurotransmitter2int_dict = {'excitatory': 0, 'inhibitory': 1, 'na': 2, 'nan': 2}
    morphology2int_dict = {'contralateral': 0, 'ipsilateral': 1}

    for work_column in columns_replace_string:
        all_cells.loc[:, work_column + "_clone"] = all_cells[work_column]
        for key in eval(f'{work_column}2int_dict').keys():
            all_cells.loc[all_cells[work_column] == key, work_column] = eval(f'{work_column}2int_dict')[key]

    # Extract labels
    labels_imaging_modality = all_cells['imaging_modality'].to_numpy()
    column_labels = list(all_cells.columns[2:-len(columns_replace_string)])

    # Extract features
    features = all_cells.iloc[:, 2:-len(columns_replace_string)].to_numpy()

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return [features, labels_imaging_modality], column_labels, all_cells


if __name__ == "__main__":
    use_k_means_classes=True
    use_new_neurotransmitter = True
    with_neg_control = False
    include_manual_em = False
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')

    #load cells, prepare and calculate metrics
    cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['pa','clem'], load_repaired=True)
    cells = prepare_data_4_metric_calc(cells,use_new_neurotransmitter,use_k_means_classes,path_to_data=path_to_data)
    calculate_metric2df(cells, 'FINAL', path_to_data, force_new=True, train_or_predict='train')

    #load preexisting metrics
    all,column_labels,all_cells = load_metrics_train('FINAL',path_to_data=path_to_data) #clem_clem_predict_pa_prediction_project_neg_controls




    #unpack metrics
    features, labels, labels_imaging_modality = all
    features_pa, labels_pa, labels_imaging_modality_pa = features[labels_imaging_modality=='photoactivation'],labels[labels_imaging_modality=='photoactivation'],labels_imaging_modality[labels_imaging_modality=='photoactivation']
    features_clem, labels_clem, labels_imaging_modality_clem = features[labels_imaging_modality=='clem'],labels[labels_imaging_modality=='clem'],labels_imaging_modality[labels_imaging_modality=='clem']

    if include_manual_em:
        manual_predicted_DTs = [173141, 131678, 133334, 141963, 146884, 153284, 168586, 175440]
        manual_predicted_DTs = [str(x) for x in manual_predicted_DTs]
        all_em, column_labels_em, all_cells_em = load_metrics_predict('FINAL', path_to_data=path_to_data)
        features_em, labels_imaging_modality = all_em
        index_of_DTs = list(all_cells_em.loc[all_cells_em['cell_name'].isin(manual_predicted_DTs),:].index)
        features_pdts,labels_imaging_modality_pdts = features_em[index_of_DTs], labels_imaging_modality[index_of_DTs]

        #add predicted DTs to main arrays
        features = np.concatenate([features,features_pdts])
        features_clem = np.concatenate([features_clem,features_pdts])
        labels = np.concatenate([labels,np.full(len(manual_predicted_DTs),'dynamic threshold')])
        labels_clem = np.concatenate([labels_clem, np.full(len(manual_predicted_DTs), 'dynamic threshold')])
        labels_imaging_modality = np.concatenate([labels_imaging_modality,np.full(len(manual_predicted_DTs),'clem')])
        labels_imaging_modality_clem = np.concatenate([labels_imaging_modality_clem, np.full(len(manual_predicted_DTs), 'clem')])



    #New segment: find best predictor
    path_to_save = path_to_data / 'make_figures_FK_output' / 'confusion_matrix'
    os.makedirs(path_to_save,exist_ok=True)

    solver='lsqr'
    for train_mod,test_mod,ftrain,ftest,ltrain,ltest, in (zip(['CLEM','PA','PA','ALL'],
                                                                                      ['CLEM','PA','CLEM',"CLEM"],
                                                                                      [features_clem,features_pa,features_pa,features],
                                                                                      [features_clem,features_pa,features_clem,features_clem],
                                                                                      [labels_clem,labels_pa,labels_pa,labels],
                                                                                      [labels_clem,labels_pa,labels_clem,labels_clem])):


        for feature_selector in ['NONE','SKB','RFC','DTC','XGBC','PI']:


            if feature_selector == 'SKB':
                reduced_features_index, no_of_featurs, evaluation_method, max_accuracy,trm,tem = select_features(ftrain, ltrain,
                                                                                                         ftest, ltest,
                                                                                                         test_mod=test_mod, train_mod=train_mod, plot=False,use_assessment_per_class=True,use_std_scale=False)
            elif feature_selector == 'NONE':
                reduced_features_index, no_of_featurs, evaluation_method, max_accuracy = np.full(ftrain.shape[1], True), np.sum(np.full(ftrain.shape[1], True)), 'All features', 'idk'
            elif feature_selector == 'RFC':
                reduced_features_index, no_of_featurs, evaluation_method, max_accuracy,trm,tem = select_features(ftrain, ltrain,
                                                                                                             ftest, ltest,
                                                                                                             test_mod=test_mod, train_mod=train_mod, plot=False,which_selection=RandomForestClassifier(n_estimators=1000),use_assessment_per_class=True,use_std_scale=False)
            elif feature_selector == 'DTC':
                reduced_features_index, no_of_featurs, evaluation_method, max_accuracy ,trm,tem = select_features(ftrain, ltrain,
                                                                                                             ftest, ltest,
                                                                                                             test_mod=test_mod, train_mod=train_mod, plot=False, which_selection=DecisionTreeClassifier(),use_assessment_per_class=True,use_std_scale=False)
            elif feature_selector == 'XGBC':
                reduced_features_index, no_of_featurs, evaluation_method, max_accuracy ,trm,tem = select_features(ftrain, ltrain,
                                                                                                             ftest, ltest,
                                                                                                             test_mod=test_mod, train_mod=train_mod, plot=False, which_selection=XGBClassifier(),use_assessment_per_class=True,use_std_scale=False)
            elif feature_selector == 'PI':
                reduced_features_index, no_of_featurs, evaluation_method, max_accuracy ,trm,tem = select_features(ftrain, ltrain,
                                                                                                             ftest, ltest,
                                                                                                             test_mod=test_mod, train_mod=train_mod, plot=False, which_selection='PI',use_assessment_per_class=True,use_std_scale=False)




            total_correct = []
            fig, ax = plt.subplots(2, 2, figsize=(30, 30))
            a = plot_prediction_matrix(features_pa[:,reduced_features_index], labels_pa,
                                   labels_imaging_modality_pa, path_to_data,
                                   np.array(column_labels)[reduced_features_index], solver = solver, shrinkage = 'auto',
                                   title = 'Trained on PA\n Tested on PA',
                                   match_limit = 0.5, return_metrics = False,
                                   ax = ax[0, 0])
            total_correct.append(a)
            a = plot_prediction_matrix_test_train_different(features_pa[:,reduced_features_index], labels_pa,
                                                        features_clem[:,reduced_features_index], labels_clem,
                                                        labels_imaging_modality_clem, path_to_data, np.array(column_labels)[reduced_features_index], train_mod_pa = True,
                                                        solver = solver, shrinkage = 'auto', title = f'Trained on PA\nTested on CLEM', match_limit = 0.5, return_metrics = False, ax = ax[0,1])
            total_correct.append(a)
            a = plot_prediction_matrix(features_clem[:,reduced_features_index], labels_clem,
                                   labels_imaging_modality_clem, path_to_data,
                                   np.array(column_labels)[reduced_features_index], solver = solver, shrinkage = 'auto',
                                   title = 'Trained on CLEM\n Tested on CLEM',
                                   match_limit = 0.5, return_metrics = False,
                                   ax = ax[1, 0])
            total_correct.append(a)
            a = plot_prediction_matrix_test_train_different(features[:,reduced_features_index], labels,
                                                        features_clem[:,reduced_features_index], labels_clem,
                                                        labels_imaging_modality_clem, path_to_data, np.array(column_labels)[reduced_features_index], train_mod_pa = False,
                                                        solver = solver, shrinkage = 'auto', title = f'Trained on PA/CLEM\nTested on CLEM', match_limit = 0.5, return_metrics = False, ax = ax[1,1])
            total_correct.append(a)
            plt.suptitle(f'Features selected {train_mod} --> {test_mod}\nSelected with: {evaluation_method}\nNo. features: {no_of_featurs}\nTotal correct: {np.round(np.mean(total_correct),2)}%',fontsize='large')
            plt.savefig(path_to_save / f'train_{train_mod}_test_{test_mod}_{evaluation_method}_{np.round(np.mean(total_correct),2)}.png')
            plt.show()

            # a = plot_prediction_matrix(features_pa[:,reduced_features_index], labels_pa,
            #                        labels_imaging_modality_pa, path_to_data,
            #                        np.array(column_labels)[reduced_features_index], solver = solver, shrinkage = 'auto',
            #                        title = 'Trained on PA\n Tested on PA',
            #                        match_limit = 0.5, return_metrics = False)
            # a = plot_prediction_matrix_test_train_different(features_pa[:,reduced_features_index], labels_pa,
            #                                             features_clem[:,reduced_features_index], labels_clem,
            #                                             labels_imaging_modality_clem, path_to_data, np.array(column_labels)[reduced_features_index], train_mod_pa = True,
            #                                             solver = solver, shrinkage = 'auto', title = f'Trained on PA\nTested on CLEM', match_limit = 0.5, return_metrics = False)
            # a = plot_prediction_matrix(features[:,reduced_features_index], labels,
            #                        labels_imaging_modality, path_to_data,
            #                        np.array(column_labels)[reduced_features_index], solver = solver, shrinkage = 'auto',
            #                        title = 'Trained on PA/CLEM\n Tested on PA/CLEM',
            #                        match_limit = 0.5, return_metrics = False)
            # a = plot_prediction_matrix_test_train_different(features[:,reduced_features_index], labels,
            #                                             features_clem[:,reduced_features_index], labels_clem,
            #                                             labels_imaging_modality_clem, path_to_data, np.array(column_labels)[reduced_features_index], train_mod_pa = False,
            #                                             solver = solver, shrinkage = 'auto', title = f'Trained on PA/CLEM\nTested on CLEM', match_limit = 0.5, return_metrics = False)



    #New segment: plot confusion matrices with best features





    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy, trm, tem = select_features(features_pa, labels_pa,
                                                                                                       features_pa, labels_pa,
                                                                                                       test_mod='PA', train_mod="PA", plot=True, which_selection=XGBClassifier(), use_assessment_per_class=True,
                                                                                                       use_std_scale=False)





    fig,ax = fig, ax = plt.subplots(2, 2, figsize=(30, 30))
    total_correct = []
    fig, ax = plt.subplots(2, 2, figsize=(30, 30))
    a = plot_prediction_matrix(features_pa[:, reduced_features_index], labels_pa,
                               labels_imaging_modality_pa, path_to_data,
                               np.array(column_labels)[reduced_features_index], solver='lsqr', shrinkage='auto',
                               title='Trained on PA\n Tested on PA',
                               match_limit=0.5, return_metrics=False,
                               ax=ax[0, 0])
    total_correct.append(a)
    a = plot_prediction_matrix_test_train_different(features_pa[:, reduced_features_index], labels_pa,
                                                    features_clem[:, reduced_features_index], labels_clem,
                                                    labels_imaging_modality_clem, path_to_data, np.array(column_labels)[reduced_features_index], train_mod_pa=True,
                                                    solver='lsqr', shrinkage='auto', title=f'Trained on PA\nTested on CLEM', match_limit=0.5, return_metrics=False, ax=ax[0, 1])
    total_correct.append(a)
    a = plot_prediction_matrix(features[:, reduced_features_index], labels,
                               labels_imaging_modality, path_to_data,
                               np.array(column_labels)[reduced_features_index], solver='lsqr', shrinkage='auto',
                               title='Trained on PA/CLEM\n Tested on PA/CLEM',
                               match_limit=0.5, return_metrics=False,
                               ax=ax[1, 0])
    total_correct.append(a)
    a = plot_prediction_matrix_test_train_different(features[:, reduced_features_index], labels,
                                                    features_clem[:, reduced_features_index], labels_clem,
                                                    labels_imaging_modality_clem, path_to_data, np.array(column_labels)[reduced_features_index], train_mod_pa=False,
                                                    solver='lsqr', shrinkage='auto', title=f'Trained on PA/CLEM\nTested on CLEM', match_limit=0.5, return_metrics=False, ax=ax[1, 1])
    total_correct.append(a)
    plt.suptitle(f'Features selected ALL --> CLEM\nSelected with: DecisionTreeClassifier\nNo. features: {no_of_featurs}\nTotal correct: {np.round(np.mean(total_correct), 2)}%', fontsize='large')
    plt.savefig(path_to_save / f'FINAL_config.png')
    plt.savefig(path_to_save / f'FINAL_config.pdf',dpi=600)
    plt.show()

    #New segment: 2d embedding
    #2d reduction

    priors = [len(labels[labels == x]) / len(labels) for x in np.unique(labels)]
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors,n_components=2)
    clf = clf.fit(features[:, reduced_features_index], labels)


    sklearn.metrics.accuracy_score(labels, clf.predict(features[:, reduced_features_index]))




    solver = 'lsqr'
    # You can choose either of these depending on your use case
    feat_transform = features
    feat_transform = features[:, reduced_features_index]

    # Fit the LDA model
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage='auto', priors=priors, n_components=2)
    clf.fit(feat_transform, labels)
    color_dict = {
        "integrator ipsilateral": '#feb326b3',
        "integrator contralateral": '#e84d8ab3',
        "dynamic threshold": '#64c5ebb3',
        "motor command": '#7f58afb3',
        'neg control': "#a8c256b3"
    }
    if solver == 'eigen':
        if clf.n_components == 2:


            X_r2 = clf.transform(feat_transform)



            X_r2 = clf.transform(feat_transform)

            plt.figure(figsize=(15, 10))
            sns.scatterplot(x=X_r2[:, 0], y=X_r2[:, 1], c=[color_dict[x] for x in labels], s=100)
            plt.title(solver + " real lables on 2d transform()")
            plt.axis('equal')
            plt.show()

            clf2 = LinearDiscriminantAnalysis(solver=solver, shrinkage='auto', priors=priors, n_components=2)
            clf2.fit(X_r2, labels)

            plt.figure(figsize=(15, 10))
            sns.scatterplot(x=X_r2[:, 0], y=X_r2[:, 1], c=[color_dict[x] for x in clf2.predict(X_r2)], s=100)
            plt.title(solver + " predicted lables on 2d transform()")
            plt.axis('equal')
            plt.show()

        elif clf.n_components == 3:
            color_dict = {
                "integrator ipsilateral": '#feb326b3',
                "integrator contralateral": '#e84d8ab3',
                "dynamic threshold": '#64c5ebb3',
                "motor command": '#7f58afb3',
                'neg control': "#a8c256b3"
            }

            X_r2 = clf.transform(feat_transform)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot data
            scatter = ax.scatter(X_r2[:, 0], X_r2[:, 1], X_r2[:, 2], c=[color_dict[x] for x in labels], marker='o')

            # Set labels
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')

            # Show the plot
            plt.show()

    sklearn.metrics.accuracy_score(labels, clf.predict(feat_transform))
    # Compute class means and overall mean
    class_means = clf.means_
    overall_mean = np.mean(feat_transform, axis=0)

    # Compute Between-Class Scatter Matrix (S_B)
    S_B = np.zeros((feat_transform.shape[1], feat_transform.shape[1]))
    for i, mean_vec in enumerate(class_means):
        n = feat_transform[labels == clf.classes_[i]].shape[0]
        mean_vec = mean_vec.reshape(-1, 1)
        overall_mean_vec = overall_mean.reshape(-1, 1)
        S_B += n * (mean_vec - overall_mean_vec).dot((mean_vec - overall_mean_vec).T)

    # Compute Within-Class Scatter Matrix (S_W)
    S_W = np.zeros((feat_transform.shape[1], feat_transform.shape[1]))
    for i, mean_vec in enumerate(class_means):
        for row in feat_transform[labels == clf.classes_[i]]:
            row = row.reshape(-1, 1)
            mean_vec = mean_vec.reshape(-1, 1)
            S_W += (row - mean_vec).dot((row - mean_vec).T)

    # Regularize S_W to avoid singular matrix error
    reg_param = 1e-6  # Small regularization parameter (can be tuned)
    S_W += reg_param * np.eye(S_W.shape[0])

    # Solve the eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # Sort eigenvectors by eigenvalues in descending order
    eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    # Select top n_components eigenvectors
    W = np.real(np.hstack([eig_pairs[i][1].reshape(-1, 1) for i in range(clf.n_components)]))

    # Transform the data using the selected eigenvectors
    X_r2 = feat_transform.dot(W)
    X_r2 = np.real(X_r2)



    sns.scatterplot(x=X_r2[:, 0], y=X_r2[:, 1], hue=clf.predict(feat_transform))
    plt.title(solver + " predicted labels")
    plt.show()

    clf3 = LinearDiscriminantAnalysis(solver=solver, shrinkage='auto', priors=priors, n_components=2)
    clf3.fit(X_r2, labels)

    savepath = path_to_data / 'make_figures_FK_output' / '2D_LDA_PROJECTION'
    os.makedirs(savepath, exist_ok=True)

    plt.figure(figsize=(15,10))
    sns.scatterplot(x=X_r2[:, 0], y=X_r2[:, 1], c=[color_dict[x] for x in clf.predict(feat_transform)],s=100)
    plt.title(solver + " predicted lables")
    plt.axis('equal')
    plt.savefig(savepath / (solver + " predicted lables own calc.pdf"),dpi=400)
    plt.show()

    plt.figure(figsize=(15,10))
    sns.scatterplot(x=X_r2[:, 0], y=X_r2[:, 1], c=[color_dict[x] for x in labels],s=100)
    plt.title(solver + " real lables")
    plt.axis('equal')
    plt.savefig(savepath / (solver + " real labels own calc.pdf"),dpi=400)
    plt.show()

    plt.figure(figsize=(15, 10))
    sns.scatterplot(x=X_r2[:, 0], y=X_r2[:, 1], c=[color_dict[x] for x in clf3.predict(X_r2)], s=100)
    plt.title(solver + " predicted lables on 2d")
    plt.savefig(savepath / (solver + " predicted lables on 2d own calc.pdf"),dpi=400)
    plt.axis('equal')


    plt.show()












    # # feature selection
    # # RANDOM FOREST
    # X, y = features[:,reduced_features_index], labels
    #
    # unique_strings = list(set(y))
    # string_to_int = {string: idx for idx, string in enumerate(y)}
    #
    # y = [string_to_int[string] for string in y]
    #
    #
    # # define the model
    # model = RandomForestClassifier().fit(X, y)
    # importance = model.feature_importances_
    #
    # #plot
    # importance_df = pd.DataFrame(np.stack([np.array(column_labels)[reduced_features_index],importance]).T).sort_values(1).reset_index(drop=True)
    # importance_df.iloc[:,1] = importance_df.iloc[:,1].astype(float)
    # plt.figure(figsize=(30,15))
    # plt.bar(importance_df.index,importance_df[1])
    # plt.xticks(np.arange(0,len(importance)),importance_df[0],rotation=45,horizontalalignment='right',verticalalignment='top')
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    # plt.show()
    # plt.figure(figsize=(30, 15))
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.xticks(np.arange(0, len(importance)), np.array(column_labels)[reduced_features_index], rotation=45, horizontalalignment='right', verticalalignment='top')
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    # plt.show()
    #
    #
    # # CART
    #
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



    #
    # #XGBOOST
    # X, y = features, labels
    #
    # unique_strings = list(set(y))
    # string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
    # y = [string_to_int[string] for string in y]
    #
    # model = XGBClassifier().fit(X, y)
    #
    # # get importance
    # importance = model.feature_importances_
    # # summarize feature importance
    #
    # # plot feature importance
    # importance_df = pd.DataFrame(np.stack([column_labels, importance]).T).sort_values(1).reset_index(drop=True)
    # importance_df.iloc[:, 1] = importance_df.iloc[:, 1].astype(float)
    # plt.figure(figsize=(30, 15))
    # plt.bar(importance_df.index, importance_df[1])
    # plt.xticks(np.arange(0, len(importance)), importance_df[0], rotation=45, horizontalalignment='right', verticalalignment='top')
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    # plt.show()
    # plt.figure(figsize=(30, 15))
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.xticks(np.arange(0, len(importance)), column_labels, rotation=45, horizontalalignment='right', verticalalignment='top')
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    # plt.show()
    #
    #
    # #PERMUTATION FEATURE IMORTANCE
    #
    # X, y = features, labels
    #
    # unique_strings = list(set(y))
    # string_to_int = {string: idx for idx, string in enumerate(unique_strings)}
    # y = [string_to_int[string] for string in y]
    #
    # model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
    # results = permutation_importance(model, X, y, scoring='accuracy',n_repeats=1000)
    # importance = results.importances_mean
    #
    # #plot
    # importance_df = pd.DataFrame(np.stack([column_labels,importance]).T)
    # importance_df.iloc[:,1] = importance_df.iloc[:,1].astype(np.float64)
    # importance_df = importance_df.sort_values(1).reset_index(drop=True)
    # plt.figure(figsize=(30,15))
    # plt.bar(importance_df.index,importance_df[1])
    # plt.xticks(np.arange(0,len(importance)),importance_df[0],rotation=45,horizontalalignment='right',verticalalignment='top')
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    # plt.show()
    # plt.figure(figsize=(30, 15))
    # plt.bar([x for x in range(len(importance))], importance)
    # plt.xticks(np.arange(0, len(importance)), column_labels, rotation=45, horizontalalignment='right', verticalalignment='top')
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.7, bottom=0.2)
    # plt.show()



    #New segment: More cross validation
    #XGBClassifier
    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy,trm,tem = select_features(features_pa, labels_pa,
                                                                                             features_pa, labels_pa,
                                                                                             test_mod='PA', train_mod="PA", plot=True, which_selection=XGBClassifier(), use_assessment_per_class=True,
                                                                                             use_std_scale=False)

    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy,trm,tem = select_features(features_pa, labels_pa,
    #                                                                                          features_clem, labels_clem,
    #                                                                                          test_mod='CLEM', train_mod="PA", plot=True, which_selection=DecisionTreeClassifier(), use_assessment_per_class=True,
    #                                                                                          use_std_scale=False)
    #
    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy,trm,tem = select_features(features_clem, labels_clem,
    #                                                                                          features_clem, labels_clem,
    #                                                                                          test_mod='CLEM', train_mod="CLEM", plot=True, which_selection=DecisionTreeClassifier(), use_assessment_per_class=True,
    #                                                                                          use_std_scale=False)


    # reduced_features_index, no_of_featurs, evaluation_method, max_accuracy,trm,tem = select_features(features, labels,
    #                                                                                          features_clem, labels_clem,
    #                                                                                          test_mod='CLEM', train_mod="ALL", plot=True, which_selection=DecisionTreeClassifier(), use_assessment_per_class=True,
    #                                                                                          use_std_scale=False)




    fig,ax = plt.subplots(2,2,figsize=(17,17))
    do_n_repeat_train_test_confusion_matrix_simple_split(features_pa[:, reduced_features_index], labels_pa, 'PA:PA',n=100,ax = ax[0,0])
    do_n_repeat_train_test_confusion_matrix_simple_split(features_clem[:, reduced_features_index], labels_clem, 'CLEM:CLEM',n=100,ax = ax[1,0])
    do_n_repeat_train_test_confusion_matrix_simple_split_test_not_train(features_pa[:, reduced_features_index], labels_pa,
                                                                        features_clem[:, reduced_features_index], labels_clem, 'PA:CLEM', test_size=0.3, n=100,ax = ax[0,1])

    do_n_repeat_train_test_confusion_matrix_simple_split_test_not_train_train_contains_test(features_pa[:, reduced_features_index], labels_pa,
                                                                        features_clem[:, reduced_features_index], labels_clem, 'ALL:CLEM', test_size=0.3, n=100,ax = ax[1,1])



    fig.suptitle(f'Features selected with {evaluation_method}\nLOOCV\nTRAIN:TEST\n{trm}:{tem}',fontsize='x-large')
    plt.show()
    from sklearn import datasets, decomposition
    pca = decomposition.PCA(n_components=10)
    pca.fit(features)

    features_pa_pca = pca.transform(features_pa)
    features_clem_pca = pca.transform(features_clem)

    fig,ax = plt.subplots(2,2,figsize=(17,17))
    do_n_repeat_train_test_confusion_matrix_simple_split(features_pa_pca, labels_pa, 'PA:PA',n=100,ax = ax[0,0])
    do_n_repeat_train_test_confusion_matrix_simple_split(features_clem_pca, labels_clem, 'CLEM:CLEM',n=100,ax = ax[1,0])
    do_n_repeat_train_test_confusion_matrix_simple_split_test_not_train(features_pa_pca, labels_pa,
                                                                        features_clem_pca, labels_clem, 'PA:CLEM', test_size=0.3, n=100,ax = ax[0,1])

    do_n_repeat_train_test_confusion_matrix_simple_split_test_not_train_train_contains_test(features_pa_pca, labels_pa,
                                                                        features_clem_pca, labels_clem, 'ALL:CLEM', test_size=0.3, n=100,ax = ax[1,1])



    fig.suptitle(f'PCA',fontsize='x-large')

    plt.show()
















    do_n_repeat_train_test_confusion_matrix_kfold(features_pa[:, reduced_features_index], labels_pa, 'pa',n_repeats=100,n_splits=5)
    do_n_repeat_train_test_confusion_matrix_kfold(features_clem[:, reduced_features_index], labels_clem, 'clem',n_repeats=100,n_splits=5)
    do_n_repeat_train_test_confusion_matrix_kfold(features[:, reduced_features_index], labels, 'all',n_repeats=100,n_splits=5)


