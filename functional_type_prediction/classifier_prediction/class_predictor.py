import sys

import numpy as np
from colorama import Fore
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.calculate_metric2df_semiold import *

np.set_printoptions(suppress=True)




class class_predictor:
    def __init__(self, path):
        self.path = path
        self.path_to_save_confusion_matrices = path / 'make_figures_FK_output' / 'all_confusion_matrices'
        os.makedirs(self.path_to_save_confusion_matrices, exist_ok=True)

    def prepare_data_4_metric_calc(self, df):
        if self.train_or_predict == 'train':
            df.loc[df['function'].isin(
                ['off-response', 'no response', 'noisy, little modulation']), 'function'] = 'neg_control'
            df.function = df.function.apply(lambda x: x.replace(' ', "_"))
            df = df.loc[(df.function != 'nan'), :]
            df = df.loc[(~df.function.isna()), :]
        df = df.drop_duplicates(keep='first', inplace=False, subset='cell_name')
        df = df.reset_index(drop=True)
        if self.train_or_predict == 'train':
            if self.kmeans_classes:
                for i, cell in df.iterrows():
                    temp_path = Path(str(cell.metadata_path)[:-4] + "_with_regressor.txt")
                    temp_path_pa = self.path / 'paGFP' / cell.cell_name / f"{cell.cell_name}_metadata_with_regressor.txt"
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
            if self.new_neurotransmitter:
                new_neurotransmitter = pd.read_excel(
                    self.path / 'em_zfish1' / 'Figures' / 'Fig 4' / 'cells2show.xlsx',
                    sheet_name='paGFP stack quality', dtype=str)

                neurotransmitter_dict = {'Vglut2a': 'excitatory', 'Gad1b': 'inhibitory'}
                for i, cell in df.iterrows():
                    if cell.imaging_modality == 'photoactivation':
                        if new_neurotransmitter.loc[new_neurotransmitter['Name'] == cell.cell_name, 'Neurotransmitter'].iloc[0] is np.nan:
                            df.loc[i, 'neurotransmitter'] = 'nan'
                        else:
                            df.loc[i, 'neurotransmitter'] = neurotransmitter_dict[new_neurotransmitter.loc[
                                new_neurotransmitter['Name'] == cell.cell_name, 'Neurotransmitter'].iloc[0]]

        return df

    def load_metrics_train(self, file_name, with_neg_control=False):
        file_path = self.path / 'prediction' / f'{file_name}_train_features.hdf5'

        all_cells = pd.read_hdf(file_path, 'complete_df')

        # throw out weird jon cells
        # all_cells = all_cells.loc[~all_cells.cell_name.isin(["cell_576460752734566521", "cell_576460752723528109", "cell_576460752684182585"]), :]

        # Data Preprocessing
        all_cells = all_cells[all_cells['function'] != 'nan']
        all_cells = all_cells.sort_values(by=['function', 'morphology', 'imaging_modality', 'neurotransmitter'])
        all_cells = all_cells.reset_index(drop=True)

        all_cells.loc[all_cells['function'].isin(
            ['no response', 'off-response', 'noisy, little modulation']), 'function'] = 'neg_control'
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

    def load_metrics_predict(self, file_name, with_neg_control=False):
        file_path = self.path / 'prediction' / f'{file_name}_predict_features.hdf5'

        all_cells = pd.read_hdf(file_path, 'complete_df')

        # throw out weird jon cells
        # all_cells = all_cells.loc[~all_cells.cell_name.isin(["cell_576460752734566521", "cell_576460752723528109", "cell_576460752684182585"]), :]

        # Data Preprocessing
        all_cells = all_cells.sort_values(by=['morphology', 'neurotransmitter'])
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

    def load_cells_features(self, file, train_or_predict='train',with_neg_control=False):
        if train_or_predict == 'train':
            self.all_train, self.column_labels_train, self.all_cells_train = self.load_metrics_train(file, with_neg_control)
            self.features_fk_train, self.labels_fk_train, self.labels_imaging_modality_train = self.all_train
        elif train_or_predict == 'predict':
            self.all_predict, self.column_labels_predict, self.all_cells_predict = self.load_metrics_predict(file, with_neg_control)
            self.features_fk_predict, self.labels_fk_predict, self.labels_imaging_modality_predict = self.all_predict

    def load_cells_df(self, kmeans_classes=True, new_neurotransmitter=True, modalities=['pa', 'clem'], train_or_predict='train'):
        self.kmeans_classes = kmeans_classes
        self.new_neurotransmitter = new_neurotransmitter
        self.modalities = modalities
        self.train_or_predict = train_or_predict

        self.cells = load_cells_predictor_pipeline(path_to_data=Path(self.path),
                                                   modalities=modalities,
                                                   load_repaired=True)

        self.cells = self.prepare_data_4_metric_calc(self.cells)
        # only select cells
        if hasattr(self, 'all_cells_train'):
            self.cells = self.cells.set_index('cell_name').loc[self.all_cells_train['cell_name']].reset_index()
        else:
            raise ValueError("Metrics have not been loaded.")
        # resample neurons 1 micron
        self.cells['swc'] = self.cells['swc'].apply(lambda x: x.resample("1 micron"))
        self.cells['class'] = self.cells.loc[:, ['function', 'morphology']].apply(
            lambda x: x['function'].replace(" ", "_") + "_" + x['morphology'] if x['function'] == 'integrator' else x['function'].replace(" ", "_"), axis=1)
        self.labels_train = self.cells['class'].to_numpy()
        self.clem_idx = (self.cells['imaging_modality'] == 'clem').to_numpy()
        self.pa_idx = (self.cells['imaging_modality'] == 'photoactivation').to_numpy()

    def calculate_published_metrics(self):
        self.features_pv = np.stack([navis.persistence_vectors(x, samples=300)[0] for x in self.cells.swc])[:, 0,
                      :]  # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184
        self.features_ps = np.stack([navis.persistence_vectors(x, samples=300)[1] for x in
                                self.cells.swc])  # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184
        self.features_ff = navis.form_factor(navis.NeuronList(self.cells.swc), n_cores=15, parallel=True,
                                        num=300)  # https://link.springer.com/article/10.1007/s12021-017-9341-1

    def select_features(self,  train_mod: str,test_mod: str, plot=False,use_assessment_per_class=False,which_selection='SKB', use_std_scale=False):

        def calc_penalty(temp_list):
            p = 2  # You can adjust this power as needed
            penalty = np.mean(temp_list) / np.exp(np.std(temp_list) ** p)
            return penalty

        def find_optimum_SKB(features_train, labels_train, features_test, labels_test, train_test_identical,
                             train_contains_test, train_mod, use_std_scale=False):



            pred_correct_dict_over_n = {}
            pred_correct_dict_over_n_per_class = {}
            used_features_idx_over_n = {}
            proba_matrix_over_n = {}
            if train_test_identical:
                for evaluator, evaluator_name in zip([f_classif, mutual_info_classif],
                                                     ['f_classif', 'mutual_info_classif']):
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

                        pred_correct_dict_over_n[evaluator_name].append(
                            np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

                        temp_list = []
                        for unique_label in np.unique(labels_test):
                            correct_in_class = np.sum(
                                [x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if
                                 x is not None])
                            percent_correct_in_class = len(
                                np.array(pred_correct_list)[np.where(labels_test == unique_label)])
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

                for evaluator, evaluator_name in zip([f_classif, mutual_info_classif],
                                                     ['f_classif', 'mutual_info_classif']):
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
                            idx_test_in_train = np.argmax(
                                np.any(np.all(X_new[:, None] == features_test[i3, idx], axis=2), axis=1))
                            features_train_without_test = X_new[
                                [x for x in range(X_new.shape[0]) if x != idx_test_in_train]]
                            labels_train_without_test = labels_train[
                                [x for x in range(X_new.shape[0]) if x != idx_test_in_train]]

                            priors = [len(labels_train_without_test[labels_train_without_test == x]) / len(
                                labels_train_without_test) for x in np.unique(labels_train_without_test)]
                            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                            clf.fit(features_train_without_test, labels_train_without_test.flatten())

                            X_test = features_test[i3, idx]
                            y_test = labels_test[i3]
                            if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                                pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                            else:
                                pred_correct_list.append(None)
                        pred_correct_dict_over_n[evaluator_name].append(
                            np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

                        temp_list = []
                        for unique_label in np.unique(labels_test):
                            correct_in_class = np.sum(
                                [x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if
                                 x is not None])
                            percent_correct_in_class = len(
                                np.array(pred_correct_list)[np.where(labels_test == unique_label)])
                            temp_list.append(correct_in_class / percent_correct_in_class)
                        if use_std_scale:
                            penalty = calc_penalty(temp_list)
                            pred_correct_dict_over_n_per_class[evaluator_name].append(penalty)
                        else:
                            pred_correct_dict_over_n_per_class[evaluator_name].append(np.mean(temp_list))

                return pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n
            else:
                for evaluator, evaluator_name in zip([f_classif, mutual_info_classif],
                                                     ['f_classif', 'mutual_info_classif']):
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

                        priors = [len(labels_train[labels_train == x]) / len(labels_train) for x in
                                  np.unique(labels_train)]
                        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
                        clf.fit(X_new, labels_train.flatten())
                        for i3 in range(features_test.shape[0]):
                            X_test = features_test[i3, idx]
                            y_test = labels_test[i3]

                            if np.max(clf.predict_proba(X_test[np.newaxis, :])) >= 0.5:
                                pred_correct_list.append((clf.predict(X_test[np.newaxis, :]) == y_test)[0])
                            else:
                                pred_correct_list.append(None)

                        pred_correct_dict_over_n[evaluator_name].append(
                            np.sum([x for x in pred_correct_list if x is not None]) / len(pred_correct_list))

                        temp_list = []
                        for unique_label in np.unique(labels_test):
                            correct_in_class = np.sum(
                                [x for x in np.array(pred_correct_list)[np.where(labels_test == unique_label)] if
                                 x is not None])
                            percent_correct_in_class = len(
                                np.array(pred_correct_list)[np.where(labels_test == unique_label)])
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

        mod2idx = {'all':np.full(len(self.pa_idx),True),'pa':self.pa_idx,'clem':self.clem_idx}

        features_train = self.features_fk_train[mod2idx[train_mod]]
        labels_train = self.labels_fk_train[mod2idx[train_mod]]
        features_test = self.features_fk_train[mod2idx[test_mod]]
        labels_test = self.labels_fk_train[mod2idx[test_mod]]
        if features_train.shape == features_test.shape:
            train_test_identical = (features_train == features_test).all()
        else:
            train_test_identical = False
        train_contains_test = np.any(np.all(features_train[:, None] == features_test, axis=2), axis=1).any()
        if which_selection == 'SKB':
            pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_SKB(
                features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test,
                train_mod, use_std_scale=use_std_scale)
        elif which_selection == "PI":
            pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_PI(
                features_train, labels_train, features_test, labels_test, train_test_identical, train_contains_test,
                train_mod, use_std_scale=use_std_scale)
        else:
            pred_correct_dict_over_n, pred_correct_dict_over_n_per_class, used_features_idx_over_n = find_optimum_custom(
                which_selection, features_train, labels_train, features_test, labels_test, train_test_identical,
                train_contains_test, train_mod, use_std_scale=use_std_scale)
        if use_assessment_per_class:
            pred_correct_dict_over_n = pred_correct_dict_over_n_per_class

        max_accuracy_idx = np.argmax([np.max(pred_correct_dict_over_n[x]) for x in pred_correct_dict_over_n.keys()])
        if len(np.unique([np.max(pred_correct_dict_over_n[x]) for x in pred_correct_dict_over_n.keys()])) == 1:
            max_accuracy_idx = np.argmin(
                [np.argmax(pred_correct_dict_over_n[x]) for x in pred_correct_dict_over_n.keys()])

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
        self.reduced_features_idx = bool_features_2_use
        self.select_train_mod = train_mod
        self.select_test_mod = test_mod
        self.select_method = str(which_selection)
        if 'XGBClassifier' in self.select_method:
            self.select_method = 'XGBClassifier'

        return bool_features_2_use, max_accuracy_key,  train_mod, test_mod

    def select_features_RFE(self,train_mod,test_mod,cv=False):
        mod2idx = {'all': np.full(len(self.pa_idx), True), 'pa': self.pa_idx, 'clem': self.clem_idx}


        from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron, PassiveAggressiveClassifier
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import BaggingClassifier
        from sklearn.feature_selection import RFE
        from sklearn.feature_selection import RFECV
        from sklearn.model_selection import StratifiedKFold, KFold

        all_estimator = [LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                      LogisticRegression(random_state=0),
                      LinearSVC(random_state=0),
                      RidgeClassifier(random_state=0),
                      Perceptron(random_state=0),
                      PassiveAggressiveClassifier(random_state=0),
                      RandomForestClassifier(random_state=0),
                      GradientBoostingClassifier(random_state=0),
                      ExtraTreesClassifier(random_state=0),
                      AdaBoostClassifier(random_state=0),
                      DecisionTreeClassifier(random_state=0)]



        if not cv:
            for estimator in all_estimator:
                acc_list = []
                for i in np.arange(1, test.features_fk_train.shape[1] + 1):
                    selector = RFE(estimator, n_features_to_select=i, step=1).fit(self.features_fk_train,self.labels_fk_train)
                    acc = self.do_cv(method='lpo',clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),feature_type='fk',
                                train_mod = train_mod,test_mod = test_mod,figure_label = str(estimator)+"_n"+str(i),fraction_across_classes=True,idx=selector.support_,plot=False)
                    acc_list.append(acc)
                plt.figure()
                plt.plot(acc_list)
                plt.axvline(np.nanargmax(acc_list), c='red', alpha=0.3)
                plt.text(np.nanargmax(acc_list) + 1, np.mean(plt.ylim()), f'n = {np.nanargmax(acc_list) + 1}', fontsize=12, color='red', ha='left', va='bottom')
                plt.gca().set_xticks(np.arange(0, self.features_fk_train.shape[1], 3), np.arange(1, self.features_fk_train.shape[1] + 2, 3))
                plt.title(f"{str(estimator)}\nAccuracy {acc_list[np.nanargmax(acc_list)]}%", fontsize='small')
                selector = RFE(estimator, n_features_to_select=np.nanargmax(acc_list) + 1, step=1).fit(self.features_fk_train, self.labels_fk_train)

                temp_str = f"Estimator_{str(estimator)}\nfeatures_{np.sum(selector.support_)}"
                temp_str = '\n'.join([x.split('(')[0] for x in temp_str.split('\n')])
                print(temp_str, '\n')

                temp_path = self.path / 'prediction' / 'RFE'
                os.makedirs(temp_path, exist_ok=True)



                le = [Patch(facecolor='white', edgecolor='white', label=x) for x in np.array(self.column_labels_train)[selector.support_]]
                plt.legend(handles=le, frameon=False, fontsize=6, loc=[1, 0.0], bbox_to_anchor=(1, 0))
                plt.subplots_adjust(left=0.1, right=0.65, top=0.80, bottom=0.1)

                temp_str = str(int(np.round(np.nanmax(acc_list)))) + "_" + temp_str.replace('\n', "_")

                plt.savefig(temp_path / f"{temp_str}.png")

                plt.show()
                pass


        elif cv:
            for estimator in all_estimator:
                for scoring in ['accuracy','balanced_accuracy','average_precision','f1','f1_weighted']:
                    for cv in [ShuffleSplit(n_splits=100, test_size=0.3, random_state=0),LeavePOut(p=1),StratifiedKFold(n_splits=5),KFold(n_splits=5)]:
                        selector = RFECV(estimator, step=1, cv=cv).fit(self.features_fk_train,self.labels_fk_train)
                        temp_str = f"Estimator_{str(estimator)}\nScoring_{str(scoring)}\nCV_{str(cv)}\nfeatures_{np.sum(selector.support_)}"
                        temp_str = '\n'.join([x.split('(')[0] for x in temp_str.split('\n')])
                        print(temp_str,'\n')
                        accuracy = self.do_cv(method='lpo', clf=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), feature_type='fk',
                                         train_mod=train_mod, test_mod=test_mod, figure_label=temp_str,
                                              fraction_across_classes=False, idx=selector.support_, plot=True)
                        temp_path = self.path / 'prediction'/'RFECV'
                        os.makedirs(temp_path,exist_ok=True)
                        temp_str = str(int(np.round(accuracy))) + "_" + temp_str.replace('\n',"_")

                        plt.savefig(temp_path/f"{temp_str}.png")
                        pass







    def do_cv(self,method: str, clf,feature_type, train_mod,test_mod, n_repeats=100, test_size=0.3, p=1, ax=None, figure_label='error:no figure label', spines_red=False,fraction_across_classes=True,idx=None,plot=True):

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


        acronym_dict = {'dynamic_threshold': "DT", 'integrator_contralateral': "CI", 'integrator_ipsilateral': "II", 'motor_command': "MC",
                        'dynamic threshold': "DT", 'integrator contralateral': "CI", 'integrator ipsilateral': "II", 'motor command': "MC"}

        mod2idx = {'all': np.full(len(self.pa_idx), True), 'pa': self.pa_idx, 'clem': self.clem_idx}

        def extract_features(feature_type, model_to_index, mode, idx):
            if feature_type == 'fk':
                return self.features_fk_train[model_to_index[mode]][:, idx]
            elif feature_type == 'pv':
                return self.features_pv[model_to_index[mode]]
            elif feature_type == 'ps':
                return self.features_ps[model_to_index[mode]]
            elif feature_type == 'ff':
                return self.features_ff[model_to_index[mode]]

            labels_train = self.labels_fk_train[mod2idx[train_mod]]
            labels_test = self.labels_fk_train[mod2idx[test_mod]]


        if idx is None:
            idx = self.reduced_features_idx
        features_train = extract_features(feature_type, mod2idx, train_mod, idx)
        features_test = extract_features(feature_type, mod2idx, test_mod, idx)
        labels_train = self.labels_fk_train[mod2idx[train_mod]]
        labels_test = self.labels_fk_train[mod2idx[test_mod]]




        if test_mod == train_mod:
            check_test_equals_train = True
        else:
            check_test_equals_train = False
        if test_mod == 'all' and train_mod!='all':
            check_train_in_test = True
            check_test_in_train = False
        elif train_mod == 'all' and test_mod!='all':
            check_train_in_test = False
            check_test_in_train = True


        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        true_labels = []
        pred_labels = []



        if method == 'lpo':
            splitter =  LeavePOut(p=p)
        elif method == 'ss':
            splitter = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=0)


        if check_test_equals_train:


            for train_index, test_index in splitter.split(features_train):
                clf_work = clone(clf)
                X_train, X_test, y_train, y_test = features_train[train_index], features_test[test_index], labels_train[train_index], labels_test[test_index]
                if check_duplicates(X_train, X_test):
                    pass
                clf_work.fit(X_train, y_train)

                try:
                    true_labels.extend(y_test)
                    pred_labels.extend(clf_work.predict(X_test))
                except:
                    pass
        elif check_test_in_train:
            lpo = LeavePOut(p=p)
            true_labels = []
            pred_labels = []
            for train_index, test_index in splitter.split(features_train):
                bool_train = np.full_like(mod2idx[test_mod],False)
                bool_test = np.full_like(mod2idx[test_mod],False)
                bool_train[train_index] = True
                bool_test[test_index] = True

                clf_work = clone(clf)
                X_train, X_test, y_train, y_test = (features_train[bool_train*mod2idx[train_mod]],
                                                    features_train[bool_test*mod2idx[test_mod]],
                                                    labels_train[bool_train*mod2idx[train_mod]],
                                                    labels_train[bool_test*mod2idx[test_mod]])
                if y_test.size != 0:

                    clf_work.fit(X_train, y_train)
                    try:
                        pred_labels.extend(clf_work.predict(X_test))
                        true_labels.extend(list(y_test))
                    except:
                        pass
        elif check_train_in_test:
            true_labels = []
            pred_labels = []
            for train_index, test_index in splitter.split(features_test):
                bool_train = np.full_like(mod2idx[test_mod],False)
                bool_test = np.full_like(mod2idx[test_mod],False)
                bool_train[train_index] = True
                bool_test[test_index] = True

                clf_work = clone(clf)
                X_train, X_test, y_train, y_test = (features_test[bool_train*mod2idx[train_mod]],
                                                    features_test[bool_test*mod2idx[test_mod]],
                                                    labels_test[bool_train*mod2idx[train_mod]],
                                                    labels_test[bool_test*mod2idx[test_mod]])
                if y_test.size != 0:

                    clf_work.fit(X_train, y_train)
                    try:
                        pred_labels.extend(clf_work.predict(X_test))
                        true_labels.extend(list(y_test))
                    except:
                        pass
        else:
            if method == 'lpo':
                for train_index, test_index in splitter.split(features_train):
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
            elif method == 'ss':
                ss_train = clone(splitter)
                ss_test = clone(splitter)
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

        if plot:
            split = f"{(1-test_size)*100}:{test_size*100}"
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 10))
                ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
                if method == "ss":
                    plt.title(f"Confusion Matrix (SS {split} x{n_repeats})" + f'\nAccuracy: {round(accuracy_score(true_labels, pred_labels) * 100, 2)}%' + f'\n{figure_label}')
                elif method == 'lpo':
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

            else:
                ConfusionMatrixDisplay(cm).plot(ax=ax, cmap='Blues')
                if method == "ss":
                    ax.title(f"Confusion Matrix (SS {split} x{n_repeats})" + f'\nAccuracy: {round(accuracy_score(true_labels, pred_labels) * 100, 2)}%' + f'\n{figure_label}')
                elif method == 'lpo':
                    ax.title(f"Confusion Matrix (LPO = {p})" + f'\nAccuracy: {round(accuracy_score(true_labels, pred_labels) * 100, 2)}%' + f'\n{figure_label}')
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


        return round(accuracy_score(true_labels, pred_labels) * 100, 2)

    def confusion_matrices(self,clf,method: str, n_repeats=100,
                           test_size=0.3, p=1,
                           fraction_across_classes=False,feature_type='fk'):

        suptitle = feature_type.upper()+"_features_"
        if method == 'lpo':
            suptitle += f'lpo{p}'
        elif method == 'ss':
            suptitle += f'ss_{int((1-test_size)*100)}_{int((test_size)*100)}'

        if feature_type == 'fk':
            suptitle += f'{self.select_train_mod}_{self.select_test_mod}_{self.select_method}'





        target_train_test = ['ALLCLEM', 'CLEMCLEM', 'PAPA']
        fig, ax = plt.subplots(3, 3, figsize=(20, 20))
        for train_mod, loc_x in zip(['ALL', "CLEM", "PA"], range(3)):
            for test_mod, loc_y in zip(['ALL', "CLEM", "PA"], range(3)):
                spines_red = train_mod + test_mod in target_train_test
                self.do_cv(method, clf,feature_type, train_mod.lower(),test_mod.lower(), figure_label=f'{train_mod}_{test_mod}',
                      ax=ax[loc_x, loc_y], spines_red=spines_red, fraction_across_classes=fraction_across_classes,n_repeats=n_repeats, test_size=test_size, p=p)
        fig.suptitle(suptitle, fontsize='xx-large')
        plt.savefig(self.path_to_save_confusion_matrices / f'{suptitle}.png')
        plt.savefig(self.path_to_save_confusion_matrices / f'{suptitle}.pdf')
        plt.show()

if __name__ == "__main__":
    # test = class_predictor(Path(r'D:\hindbrain_structure_function\nextcloud'))
    # test.load_cells_features('FINAL_before_last_upload')
    # test.load_cells_df()
    # # test.calculate_published_metrics()
    # reduced_features_index, evaluation_method, trm, tem = test.select_features('all','clem',which_selection=DecisionTreeClassifier())
    # rfidx = reduced_features_index

    test = class_predictor(Path(r'D:\hindbrain_structure_function\nextcloud'))
    test.load_cells_features('FINAL')
    test.load_cells_df()
    # test.calculate_published_metrics()
    test.select_features_RFE('all','clem',cv=True)



    reduced_features_index, evaluation_method, trm, tem = test.select_features('all', 'clem', which_selection=XGBClassifier(), plot=True, use_std_scale=False, use_assessment_per_class=False)


    clf_fk = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    n_estimators_rf = 100
    clf_pv = RandomForestClassifier(n_estimators=n_estimators_rf)
    clf_ps = RandomForestClassifier(n_estimators=n_estimators_rf)
    clf_ff = RandomForestClassifier(n_estimators=n_estimators_rf)

    test.confusion_matrices(clf_fk,method='ss')
    print(len(np.array(test.column_labels_train)[test.reduced_features_idx]))
    print(np.array(test.column_labels_train)[test.reduced_features_idx])
    pass
