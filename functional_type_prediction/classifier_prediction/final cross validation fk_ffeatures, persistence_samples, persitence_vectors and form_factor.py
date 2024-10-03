import sys
from colorama import Fore
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.calculate_metric2df_semiold import *
np.set_printoptions(suppress=True)

def load_metrics_train(file_name, path_to_data, with_neg_control=False):
    file_path = path_to_data / 'make_figures_FK_output' / f'{file_name}_train_features.hdf5'

    all_cells = pd.read_hdf(file_path, 'complete_df')

    # throw out weird jon cells
    # all_cells = all_cells.loc[~all_cells.cell_name.isin(["cell_576460752734566521", "cell_576460752723528109", "cell_576460752684182585"]), :]

    # Data Preprocessing
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

    return [features, labels, labels_imaging_modality], column_labels, all_cells

def calc_penalty(temp_list):
    p = 2  # You can adjust this power as needed
    penalty = np.mean(temp_list) / np.exp(np.std(temp_list) ** p)
    return penalty

def find_optimum_custom(custom_scorer, features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test, train_mod, use_std_scale=False):
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

                if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                    pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                else:
                    pred_correct_list.append(None)

            pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

            temp_list = []
            for unique_label in np.unique(labels_test):
                correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if x is not None])
                percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test == unique_label)])
                temp_list.append(correct_in_class / percent_correct_in_class)
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

            X_new = features_train[:, idx]
            for i3 in range(features_test.shape[0]):
                idx_test_in_train = np.argmax(np.any(np.all(X_new[:, None] == features_test[i3, idx], axis=2), axis=1))
                features_train_without_test = X_new[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]
                labels_train_without_test = labels_train[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]

                priors = [len(labels_train_without_test[labels_train_without_test == x]) / len(labels_train_without_test) for x in np.unique(labels_train_without_test)]
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                clf.fit(features_train_without_test, labels_train_without_test.flatten())

                X_test = features_test[i3, idx]
                y_test = labels_test[i3]
                if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                    pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                else:
                    pred_correct_list.append(None)
            pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

            temp_list = []
            for unique_label in np.unique(labels_test):
                correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if x is not None])
                percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test == unique_label)])
                temp_list.append(correct_in_class / percent_correct_in_class)
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

                if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                    pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                else:
                    pred_correct_list.append(None)

            pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

            temp_list = []
            for unique_label in np.unique(labels_test):
                correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if x is not None])
                percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test == unique_label)])
                temp_list.append(correct_in_class / percent_correct_in_class)
            if use_std_scale:
                penalty = calc_penalty(temp_list)
                pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
            else:
                pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))

        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n

def find_optimum_PI(features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test, train_mod, use_std_scale=False):
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

            idx = importance >= importance[np.argsort(importance, axis=0)[::-1][no_features - 1]]

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

                if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                    pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                else:
                    pred_correct_list.append(None)

            pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

            temp_list = []
            for unique_label in np.unique(labels_test):
                correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if x is not None])
                percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test == unique_label)])
                temp_list.append(correct_in_class / percent_correct_in_class)
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
            idx = importance >= importance[np.argsort(importance, axis=0)[::-1][no_features - 1]]

            used_features_idx_over_n[evaluator_name][no_features] = idx
            pred_correct_list = []

            X_new = features_train[:, idx]
            for i3 in range(features_test.shape[0]):
                idx_test_in_train = np.argmax(np.any(np.all(X_new[:, None] == features_test[i3, idx], axis=2), axis=1))
                features_train_without_test = X_new[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]
                labels_train_without_test = labels_train[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]

                priors = [len(labels_train_without_test[labels_train_without_test == x]) / len(labels_train_without_test) for x in np.unique(labels_train_without_test)]
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                clf.fit(features_train_without_test, labels_train_without_test.flatten())

                X_test = features_test[i3, idx]
                y_test = labels_test[i3]
                if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                    pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                else:
                    pred_correct_list.append(None)
            pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

            temp_list = []
            for unique_label in np.unique(labels_test):
                correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if x is not None])
                percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test == unique_label)])
                temp_list.append(correct_in_class / percent_correct_in_class)
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

                if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                    pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                else:
                    pred_correct_list.append(None)

            pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

            temp_list = []
            for unique_label in np.unique(labels_test):
                correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if x is not None])
                percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test == unique_label)])
                temp_list.append(correct_in_class / percent_correct_in_class)
            if use_std_scale:
                penalty = calc_penalty(temp_list)
                pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
            else:
                pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))

        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n

def find_optimum_SKB(features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test, train_mod, use_std_scale=False):
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

                    if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                        pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                    else:
                        pred_correct_list.append(None)

                pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

                temp_list = []
                for unique_label in np.unique(labels_test):
                    correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if x is not None])
                    percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test == unique_label)])
                    temp_list.append(correct_in_class / percent_correct_in_class)
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

                X_new = features_train[:, idx]
                for i3 in range(features_test.shape[0]):
                    idx_test_in_train = np.argmax(np.any(np.all(X_new[:, None] == features_test[i3, idx], axis=2), axis=1))
                    features_train_without_test = X_new[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]
                    labels_train_without_test = labels_train[[x for x in range(X_new.shape[0]) if x != idx_test_in_train]]

                    priors = [len(labels_train_without_test[labels_train_without_test == x]) / len(labels_train_without_test) for x in np.unique(labels_train_without_test)]
                    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                    clf.fit(features_train_without_test, labels_train_without_test.flatten())

                    X_test = features_test[i3, idx]
                    y_test = labels_test[i3]
                    if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                        pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                    else:
                        pred_correct_list.append(None)
                pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

                temp_list = []
                for unique_label in np.unique(labels_test):
                    correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if x is not None])
                    percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test == unique_label)])
                    temp_list.append(correct_in_class / percent_correct_in_class)
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

                    if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                        pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                    else:
                        pred_correct_list.append(None)

                pred_correct_dict_over_n[evaluator_name].append(np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

                temp_list = []
                for unique_label in np.unique(labels_test):
                    correct_in_class = np.sum([x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if x is not None])
                    percent_correct_in_class = len(np.array(pred_correct_list)[np.where(labels_test == unique_label)])
                    temp_list.append(correct_in_class / percent_correct_in_class)
                if use_std_scale:
                    penalty = calc_penalty(temp_list)
                    pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
                else:
                    pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))

        return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n

def prepare_data_4_metric_calc(df, use_new_neurotransmitter, use_k_means_classes, path_to_data, train_or_predict='train'):
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

def select_features(features_train: np.ndarray, labels_train: np.ndarray, features_test: np.ndarray, labels_test: np.ndarray, test_mod: str, train_mod: str, plot=False, use_assessment_per_class=False,
                    which_selection='SKB', use_std_scale=False):
    if features_train.shape == features_test.shape:
        train_test_identical = (features_train == features_test).all()
    else:
        train_test_identical = False
    train_contains_test = np.any(np.all(features_train[:, None] == features_test, axis=2), axis=1).any()
    if which_selection == 'SKB':
        pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_SKB(features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test,
                                                                                                                  train_mod, use_std_scale=use_std_scale)
    elif which_selection == "PI":
        pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_PI(features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test,
                                                                                                                 train_mod, use_std_scale=use_std_scale)
    else:
        pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_custom(which_selection, features_train, labels_train, features_test, labels_test, train_test_identical,
                                                                                                                     train_contains_test, train_mod, use_std_scale=use_std_scale)
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

    return bool_features_2_use, max_accuracy_no_of_feat, max_accuracy_key, max_accuracy, train_mod, test_mod

def check_duplicates(train, test):
    # Convert both arrays to sets of rows
    train_set = set(map(tuple, train))
    test_set = set(map(tuple, test))

    # Find common rows between Train and Test
    common_rows = train_set.intersection(test_set)

    if common_rows:
        print(f"DEBUG: {len(common_rows)} duplicate rows found between Train and Test.")
        return True
    else:
        pass
        return False
def do_cv(method: str, clf, features_train, labels_train, features_test, labels_test, n_repeats=100, test_size=0.3, p=1, ax=None, figure_label='error:no figure label', spines_red=False,fraction_across_classes=True):
        acronym_dict = {'dynamic_threshold': "DT", 'integrator_contralateral': "CI", 'integrator_ipsilateral': "II", 'motor_command': "MC"}

        try:
            check_test_equals_train = (features_train == features_test).all()
        except:
            check_test_equals_train = False
        check_train_in_test = np.isin(features_train, features_test).all()
        check_test_in_train = np.isin(features_test, features_train).all()

        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        true_labels = []
        pred_labels = []

        if method == 'ss':
            if check_test_equals_train:
                ss = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=0)
                for train_index, test_index in ss.split(features_train):
                    clf_work = clone(clf)
                    X_train, X_test, y_train, y_test = features_train[train_index], features_test[test_index], labels_train[train_index], labels_test[test_index]
                    if check_duplicates(X_train, X_test):
                        pass
                    clf_work.fit(X_train, y_train)

                    try:
                        pred_labels.extend(clf_work.predict(X_test))
                        true_labels.extend(y_test)


                    except:

                        pass
                if fraction_across_classes:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='true').astype(float)
                else:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='pred').astype(float)

            elif check_test_in_train:
                ss = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=0)
                for train_index, test_index in ss.split(features_train):
                    clf_work = clone(clf)
                    X_train, X_test, y_train, y_test = features_train[train_index], features_train[test_index], labels_train[train_index], labels_train[test_index]

                    bool_test = np.any(np.all(features_test[:, None] == X_test, axis=2), axis=1)
                    X_test, y_test = features_test[bool_test], labels_test[bool_test]
                    if check_duplicates(X_train, X_test):
                        pass

                    clf_work.fit(X_train, y_train)

                    try:
                        pred_labels.extend(clf_work.predict(X_test))
                        true_labels.extend(list(y_test))
                    except:
                        pass

                if fraction_across_classes:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='true').astype(float)
                else:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='pred').astype(float)
            elif check_train_in_test:
                ss = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=0)
                for train_index, test_index in ss.split(features_test):
                    clf_work = clone(clf)
                    X_train, X_test, y_train, y_test = features_test[train_index], features_test[test_index], labels_test[train_index], labels_test[test_index]

                    bool_train = np.any(np.all(features_train[:, None] == X_train, axis=2), axis=1)
                    X_train, y_train = features_train[bool_train], labels_train[bool_train]

                    # Example usage
                    if check_duplicates(X_train, X_test):
                        pass

                    clf_work.fit(X_train, y_train)

                    try:
                        pred_labels.extend(clf_work.predict(X_test))
                        true_labels.extend(y_test)
                    except:
                        pass
                if fraction_across_classes:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='true').astype(float)
                else:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='pred').astype(float)
            else:
                ss_train = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=0)
                ss_test = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=0)
                for train_indeces, test_indeces in zip(ss_train.split(features_train), ss_test.split(features_test)):
                    clf_work = clone(clf)
                    train_index = train_indeces[0]
                    test_index = test_indeces[1]
                    X_train, X_test, y_train, y_test = features_train[train_index], features_test[test_index], labels_train[train_index], labels_test[test_index]
                    if check_duplicates(X_train, X_test):
                        pass

                    clf_work.fit(X_train, y_train)

                    try:
                        pred_labels.extend(clf_work.predict(X_test))
                        true_labels.extend(y_test)
                    except:
                        pass

                if fraction_across_classes:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='true').astype(float)
                else:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='pred').astype(float)
            split = f'{int((1 - test_size) * 100)}:{int((test_size) * 100)}'
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 10))
                ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
                plt.title(f"Confusion Matrix (SS {split} x{n_repeats})" + f'\nAccuracy: {round(accuracy_score(true_labels, pred_labels) * 100, 2)}%' + f'\n{figure_label}')
                ax.set_xticklabels([acronym_dict[x] for x in clf_work.classes_])
                ax.set_yticklabels([acronym_dict[x] for x in clf_work.classes_])
                if spines_red:
                    ax.spines['bottom'].set_color('red')
                    ax.spines['top'].set_color('red')
                    ax.spines['left'].set_color('red')
                    ax.spines['right'].set_color('red')
                    ax.spines['bottom'].set_linewidth(2)
                    ax.spines['top'].set_linewidth(2)
                    ax.spines['left'].set_linewidth(2)
                    ax.spines['right'].set_linewidth(2)
                plt.show()
            else:
                ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
                ax.title.set_text(f"Confusion Matrix (SS {split} x{n_repeats})" + f'\nAccuracy: {round(accuracy_score(true_labels, pred_labels) * 100, 2)}%' + f'\n{figure_label}')
                ax.set_xticklabels([acronym_dict[x] for x in clf_work.classes_])
                ax.set_yticklabels([acronym_dict[x] for x in clf_work.classes_])
                if spines_red:
                    ax.spines['bottom'].set_color('red')
                    ax.spines['top'].set_color('red')
                    ax.spines['left'].set_color('red')
                    ax.spines['right'].set_color('red')
                    ax.spines['bottom'].set_linewidth(2)
                    ax.spines['top'].set_linewidth(2)
                    ax.spines['left'].set_linewidth(2)
                    ax.spines['right'].set_linewidth(2)
        if method == 'lpo':
            if check_test_equals_train:
                lpo = LeavePOut(p=p)
                for train_index, test_index in lpo.split(features_train):
                    clf_work = clone(clf)
                    X_train, X_test, y_train, y_test = features_train[train_index], features_test[test_index], labels_train[train_index], labels_test[test_index]
                    true_labels.extend(y_test)
                    if check_duplicates(X_train, X_test):
                        pass

                    clf_work.fit(X_train, y_train)

                    pred_labels.extend(clf_work.predict(X_test))

                if fraction_across_classes:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='true').astype(float)
                else:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='pred').astype(float)
            elif check_test_in_train:
                lpo = LeavePOut(p=p)
                for train_index, test_index in lpo.split(features_train):
                    clf_work = clone(clf)
                    X_train, X_test, y_train, y_test = features_train[train_index], features_train[test_index], labels_train[train_index], labels_train[test_index]

                    bool_test = ~np.any(np.all(features_test[:, None] == X_train, axis=2), axis=1)
                    X_test, y_test = features_test[bool_test], labels_test[bool_test]
                    if check_duplicates(X_train, X_test):
                        pass

                    clf_work.fit(X_train, y_train)
                    try:
                        pred_labels.extend(clf_work.predict(X_test))
                        true_labels.extend(list(y_test))
                    except:
                        pass

                if fraction_across_classes:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='true').astype(float)
                else:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='pred').astype(float)
            elif check_train_in_test:
                lpo = LeavePOut(p=p)
                for train_index, test_index in lpo.split(features_train):
                    clf_work = clone(clf)
                    X_train, X_test, y_train, y_test = features_train[train_index], features_train[test_index], labels_train[train_index], labels_train[test_index]

                    bool_test = ~np.any(np.all(features_train[:, None] == X_test, axis=2), axis=1)
                    X_train, y_train = features_train[bool_test], labels_train[bool_test]

                    if check_duplicates(X_train, X_test):
                        pass
                    clf_work.fit(X_train, y_train)
                    try:
                        pred_labels.extend(clf_work.predict(X_test))
                        true_labels.extend(list(y_test))
                    except:
                        pass
                if fraction_across_classes:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='true').astype(float)
                else:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='pred').astype(float)
            else:
                lpo = LeavePOut(p=p)
                for train_index, test_index in lpo.split(features_train):
                    clf_work = clone(clf)
                    X_train, X_test, y_train, y_test = features_train[train_index], features_test, labels_train[train_index], labels_test

                    if check_duplicates(X_train, X_test):
                        pass
                    clf_work.fit(X_train, y_train)
                    try:
                        pred_labels.extend(clf_work.predict(X_test))
                        true_labels.extend(list(y_test))
                    except:
                        pass
                if fraction_across_classes:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='true').astype(float)
                else:
                    cm = confusion_matrix(true_labels, pred_labels, normalize='pred').astype(float)
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 10))
                ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
                plt.title(f"Confusion Matrix (LPO = {p})" + f'\nAccuracy: {round(accuracy_score(true_labels, pred_labels) * 100, 2)}%' + f'\n{figure_label}')
                ax.set_xticklabels([acronym_dict[x] for x in clf_work.classes_])
                ax.set_yticklabels([acronym_dict[x] for x in clf_work.classes_])
                if spines_red:
                    ax.spines['bottom'].set_color('red')
                    ax.spines['top'].set_color('red')
                    ax.spines['left'].set_color('red')
                    ax.spines['right'].set_color('red')
                    ax.spines['bottom'].set_linewidth(2)
                    ax.spines['top'].set_linewidth(2)
                    ax.spines['left'].set_linewidth(2)
                    ax.spines['right'].set_linewidth(2)
                plt.show()
            else:
                ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
                ax.title.set_text(f"Confusion Matrix (LPO = {p})" + f'\nAccuracy: {round(accuracy_score(true_labels, pred_labels) * 100, 2)}%' + f'\n{figure_label}')
                ax.set_xticklabels([acronym_dict[x] for x in clf_work.classes_])
                ax.set_yticklabels([acronym_dict[x] for x in clf_work.classes_])
                if spines_red:
                    ax.spines['bottom'].set_color('red')
                    ax.spines['top'].set_color('red')
                    ax.spines['left'].set_color('red')
                    ax.spines['right'].set_color('red')
                    ax.spines['bottom'].set_linewidth(2)
                    ax.spines['top'].set_linewidth(2)
                    ax.spines['left'].set_linewidth(2)
                    ax.spines['right'].set_linewidth(2)

get_gb = lambda x: sys.getsizeof(x) / (1024 ** 3)
if __name__ == "__main__":
    # New segment: set constants
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')
    path_to_save = path_to_data / 'make_figures_FK_output' / 'all_confusion_matrices'
    os.makedirs(path_to_save, exist_ok=True)
    n_estimators_rf = 100
    use_new_neurotransmitter = True
    use_k_means_classes = True
    fraction_across_classes = False

    # New segment: load FK_features
    all, column_labels, all_cells = load_metrics_train('FINAL', path_to_data=path_to_data)  # clem_clem_predict_pa_prediction_project_neg_controls

    # unpack metrics
    features_fk, labels_fk, labels_imaging_modality = all

    # New segment: load all cells and calculate persistence_vectors, persistence_samples, form_factor
    cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['pa', 'clem'], load_repaired=True)
    cells = prepare_data_4_metric_calc(cells, use_new_neurotransmitter, use_k_means_classes, path_to_data=path_to_data)
    cells = cells.set_index('cell_name').loc[all_cells['cell_name']].reset_index()
    cells['swc'] = cells['swc'].apply(lambda x: x.resample("1 micron"))



    features_pv = np.stack([navis.persistence_vectors(x, samples=300)[0] for x in cells.swc])[:, 0, :]  # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184
    features_ps = np.stack([navis.persistence_vectors(x, samples=300)[1] for x in cells.swc])  # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184
    features_ff = navis.form_factor(navis.NeuronList(cells.swc), n_cores=15, parallel=True, num=300)  # https://link.springer.com/article/10.1007/s12021-017-9341-1

    cells['class'] = cells.loc[:, ['function', 'morphology']].apply(lambda x: x['function'].replace(" ", "_") + "_" + x['morphology'] if x['function'] == 'integrator' else x['function'].replace(" ", "_"), axis=1)
    labels = cells['class'].to_numpy()

    #subset to specific classes
    # cells = cells.loc[cells['class'].isin(['integrator_contralateral','motor_command']),:]
    # features_fk = features_fk[(labels_fk=='motor command')|(labels_fk=='integrator contralateral')]
    # labels = labels[(labels_fk == 'motor command') | (labels_fk == 'integrator contralateral')]
    # labels_fk = labels_fk[(labels_fk=='motor command')|(labels_fk=='integrator contralateral')]


    clem_idx = (cells['imaging_modality'] == 'clem').to_numpy()
    pa_idx = (cells['imaging_modality'] == 'photoactivation').to_numpy()

    clf_fk = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_pv = RandomForestClassifier(n_estimators=n_estimators_rf)
    clf_ps = RandomForestClassifier(n_estimators=n_estimators_rf)
    clf_ff = RandomForestClassifier(n_estimators=n_estimators_rf)



    target_train_test = ['ALLCLEM', 'CLEMCLEM', 'PAPA']

    # #New segment: FK features, selected with PA:PA
    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy, trm, tem = select_features(features_fk[pa_idx], labels[pa_idx],
                                                                                                       features_fk[pa_idx], labels[pa_idx],
                                                                                                       test_mod='PA', train_mod="PA", plot=False, which_selection=DecisionTreeClassifier(), use_assessment_per_class=True,
                                                                                                       use_std_scale=False)

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('lpo', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_pa_pa_lpo1.png')
    plt.savefig(path_to_save / 'FK_features_pa_pa_lpo1.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_pa_pa_ss70_30_dtc.png')
    plt.savefig(path_to_save / 'FK_features_pa_pa_ss70_30_dtc.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], test_size=0.01, n_repeats=1000, spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_pa_pa_ss99_1_dtc.png')
    plt.savefig(path_to_save / 'FK_features_pa_pa_ss99_1_dtc.pdf')
    plt.show()

    print(Fore.RED + 'FINISHED FK features, selected with PA:PA DTC' + Fore.BLACK)


    # New segment: FK features, selected with PA:PA
    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy, trm, tem = select_features(features_fk[pa_idx], labels[pa_idx],
                                                                                                       features_fk[pa_idx], labels[pa_idx],
                                                                                                       test_mod='PA', train_mod="PA", plot=False, which_selection=XGBClassifier(), use_assessment_per_class=True,
                                                                                                       use_std_scale=False)

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('lpo', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_pa_pa_lpo1_xgb.png')
    plt.savefig(path_to_save / 'FK_features_pa_pa_lpo1_xgb.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_pa_pa_ss70_30_xgb.png')
    plt.savefig(path_to_save / 'FK_features_pa_pa_ss70_30_xgb.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], test_size=0.01, n_repeats=1000, spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_pa_pa_ss99_1_xgb.png')
    plt.savefig(path_to_save / 'FK_features_pa_pa_ss99_1_xgb.pdf')
    plt.show()

    print(Fore.RED + 'FINISHED FK features, selected with PA:PA XGB' + Fore.BLACK)

    # New segment: FK features, selected with ALL:CLEM
    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy, trm, tem = select_features(features_fk, labels,
                                                                                                       features_fk[clem_idx], labels[clem_idx],
                                                                                                       test_mod='CLEM', train_mod="ALL", plot=False, which_selection=DecisionTreeClassifier(),
                                                                                                       use_assessment_per_class=True,
                                                                                                       use_std_scale=False)

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('lpo', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_all_clem_lpo_dtc.png')
    plt.savefig(path_to_save / 'FK_features_all_clem_lpo_dtc.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_all_clem_ss70_30_dtc.png')
    plt.savefig(path_to_save / 'FK_features_all_clem_ss70_30_dtc.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], test_size=0.01, n_repeats=1000, spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_all_clem_ss99_1_dtc.png')
    plt.savefig(path_to_save / 'FK_features_all_clem_ss99_1_dtc.pdf')
    plt.show()
    print(Fore.RED + 'FINISHED FK features, selected with ALL:CLEM DTC' + Fore.BLACK)

    # New segment: FK features, selected with CLEM:CLEM
    reduced_features_index, no_of_featurs, evaluation_method, max_accuracy, trm, tem = select_features(features_fk[clem_idx], labels[clem_idx],
                                                                                                       features_fk[clem_idx], labels[clem_idx],
                                                                                                       test_mod='CLEM', train_mod="CLEM", plot=False, which_selection=DecisionTreeClassifier(),
                                                                                                       use_assessment_per_class=True,
                                                                                                       use_std_scale=False)

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('lpo', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_clem_clem_lpo1.png')
    plt.savefig(path_to_save / 'FK_features_clem_clem_lpo1.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y])
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_clem_clem_ss70_30.png')
    plt.savefig(path_to_save / 'FK_features_clem_clem_ss70_30.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_fk.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_fk, features_fk[:, reduced_features_index][train_idx], labels[train_idx], features_fk[:, reduced_features_index][test_idx], labels[test_idx],
                  figure_label=f'FK features\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], test_size=0.01, n_repeats=1000, spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Feature selected {evaluation_method}\nTEST:TRAIN\n{trm}:{tem}',fontsize='xx-large')
    plt.savefig(path_to_save / 'FK_features_clem_clem_ss99_1.png')
    plt.savefig(path_to_save / 'FK_features_clem_clem_ss99_1.pdf')
    plt.show()

    print(Fore.RED + 'FINISHED FK features, selected with CLEM:CLEM DTC' + Fore.BLACK)


    # New segment: Persistence Sample (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184)

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_ps.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_ps.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('lpo', clf_ps, features_ps[train_idx], labels[train_idx], features_ps[test_idx], labels[test_idx], figure_label=f'Persistence samples\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y],
                  spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Persistence samples Li et al. 2017',fontsize='xx-large')
    plt.savefig(path_to_save / 'PS_lpo1.png')
    plt.savefig(path_to_save / 'PS_lpo1.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_ps.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_ps.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_ps, features_ps[train_idx], labels[train_idx], features_ps[test_idx], labels[test_idx], figure_label=f'Persistence samples\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Persistence samples Li et al. 2017',fontsize='xx-large')
    plt.savefig(path_to_save / 'PS_ss70_30.png')
    plt.savefig(path_to_save / 'PS_ss70_30.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_ps.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_ps.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_ps, features_ps[train_idx], labels[train_idx], features_ps[test_idx], labels[test_idx], figure_label=f'Persistence samples\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], test_size=0.01,
                  n_repeats=1000, spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Persistence samples Li et al. 2017',fontsize='xx-large')
    plt.savefig(path_to_save / 'PS_ss99_1.png')
    plt.savefig(path_to_save / 'PS_ss99_1.pdf')
    plt.show()

    print(Fore.RED + 'FINISHED persistence samples' + Fore.BLACK)

    # New segment: Persistence Vector (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184)

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_pv.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_pv.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('lpo', clf_pv, features_pv[train_idx], labels[train_idx], features_pv[test_idx], labels[test_idx], figure_label=f'Persistence vectors\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y],
                  spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Persistence vectors Li et al. 2017',fontsize='xx-large')
    plt.savefig(path_to_save / 'PV_lpo1.png')
    plt.savefig(path_to_save / 'PV_lpo1.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_pv.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_pv.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_pv, features_pv[train_idx], labels[train_idx], features_pv[test_idx], labels[test_idx], figure_label=f'Persistence vectors\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Persistence vectors Li et al. 2017',fontsize='xx-large')
    plt.savefig(path_to_save / 'PV_ss70_30.png')
    plt.savefig(path_to_save / 'PV_ss70_30.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_pv.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_pv.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_pv, features_pv[train_idx], labels[train_idx], features_pv[test_idx], labels[test_idx], figure_label=f'Persistence vectors\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], test_size=0.01,
                  n_repeats=1000, spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Persistence vectors Li et al. 2017',fontsize='xx-large')
    plt.savefig(path_to_save / 'PV_ss99_1.png')
    plt.savefig(path_to_save / 'PV_ss99_1.pdf')
    plt.show()
    print(Fore.RED + 'FINISHED persistence vectors' + Fore.BLACK)

    # New segment: Form factor (https://link.springer.com/article/10.1007/s12021-017-9341-1)

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_ff.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_ff.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('lpo', clf_ff, features_ff[train_idx], labels[train_idx], features_ff[test_idx], labels[test_idx], figure_label=f'Form factors\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Form factors Choi et al. 2022',fontsize='xx-large')
    plt.savefig(path_to_save / 'FF_lpo1.png')
    plt.savefig(path_to_save / 'FF_lpo1.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_ff.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_ff.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_ff, features_ff[train_idx], labels[train_idx], features_ff[test_idx], labels[test_idx], figure_label=f'Form factors\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Form factors Choi et al. 2022',fontsize='xx-large')
    plt.savefig(path_to_save / 'FF_ss70_30.png')
    plt.savefig(path_to_save / 'FF_ss70_30.pdf')
    plt.show()

    fig, ax = plt.subplots(3, 3, figsize=(20, 20))
    for train_idx, train_mod, loc_x in zip([np.full(features_ff.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
        for test_idx, test_mod, loc_y in zip([np.full(features_ff.shape[0], True), clem_idx, pa_idx], ['ALL', "CLEM", "PA"], range(3)):
            spines_red = train_mod + test_mod in target_train_test
            do_cv('ss', clf_ff, features_ff[train_idx], labels[train_idx], features_ff[test_idx], labels[test_idx], figure_label=f'Form factors\n{train_mod}:{test_mod}', ax=ax[loc_x, loc_y], test_size=0.01,
                  n_repeats=1000, spines_red=spines_red,fraction_across_classes=fraction_across_classes)
    fig.suptitle(f'Form factors Choi et al. 2022',fontsize='xx-large')
    plt.savefig(path_to_save / 'FF_ss99_1.png')
    plt.savefig(path_to_save / 'FF_ss99_1.pdf')
    plt.show()
    print(Fore.RED + 'FINISHED form factors' + Fore.BLACK)
