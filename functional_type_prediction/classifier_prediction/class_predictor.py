import os

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
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
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import plotly
from sklearn.ensemble import IsolationForest
import pickle
from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM
from colorama import Fore, Style
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.calculate_metric2df import *
np.set_printoptions(suppress=True)
from hindbrain_structure_function.functional_type_prediction.NBLAST.nblast_matrix_navis import *
from slack_sdk import WebClient

import getpass

def send_slack_message(RECEIVER="Florian Kämpf",MESSAGE="Script finished!"):
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

class class_predictor:
    """
    A class to handle the loading, processing, and analysis of cell metrics data.

    This class provides methods to load cell metrics from a file, preprocess the data, calculate various metrics,
    and select optimal features for machine learning models. It also includes methods for cross-validation,
    confusion matrix generation, and prediction of cell types.

    Attributes:
    -----------
    features_fk_with_to_predict : numpy.ndarray
        Loaded features from the file, including 'to_predict' and 'neg_control' categories.
    labels_fk_with_to_predict : numpy.ndarray
        Loaded labels from the file, including 'to_predict' and 'neg_control' categories.
    labels_imaging_modality_with_to_predict : numpy.ndarray
        Loaded imaging modality labels from the file, including 'to_predict' and 'neg_control' categories.
    labels_fk : numpy.ndarray
        Filtered labels excluding 'to_predict' and 'neg_control' categories.
    features_fk : numpy.ndarray
        Filtered features excluding 'to_predict' and 'neg_control' categories.
    labels_imaging_modality : numpy.ndarray
        Filtered imaging modality labels excluding 'to_predict' and 'neg_control' categories.
    all_cells : pandas.DataFrame
        DataFrame containing all cells excluding 'to_predict' and 'neg_control' categories.
    cells : pandas.DataFrame
        DataFrame containing all cells with 'to_predict' attribute, present in the loaded file.
    clem_idx : numpy.ndarray
        Indicator array for cells with 'clem' imaging modality.
    pa_idx : numpy.ndarray
        Indicator array for cells with 'photoactivation' imaging modality.
    em_idx : numpy.ndarray
        Indicator array for cells with 'EM' imaging modality.
    clem_idx_with_to_predict : numpy.ndarray
        Indicator array for cells with 'clem' imaging modality, including 'to_predict' category.
    pa_idx_with_to_predict : numpy.ndarray
        Indicator array for cells with 'photoactivation' imaging modality, including 'to_predict' category.
    em_idx_with_to_predict : numpy.ndarray
        Indicator array for cells with 'EM' imaging modality, including 'to_predict' category.

    Methods:
    --------
    load_metrics(file, with_neg_control):
        Loads the metrics from the specified file.
    load_cells_df(kmeans_classes=True, new_neurotransmitter=True, modalities=['pa', 'clem'], neg_control=True):
        Loads the cells DataFrame and applies the preprocessing pipeline to the data.
    calculate_published_metrics():
        Calculates published metrics over the cell data.
    select_features(train_mod, test_mod, plot=False, use_assessment_per_class=False, which_selection='SKB', use_std_scale=False):
        Selects the optimal features for a given training and testing modality.
    select_features_RFE(train_mod, test_mod, cv=False, estimator=None, scoring=None, cv_method=None, save_features=False):
        Selects features using Recursive Feature Elimination (RFE) with cross-validation.
    do_cv(method, clf, feature_type, train_mod, test_mod, n_repeats=100, test_size=0.3, p=1, ax=None, figure_label='error:no figure label', spines_red=False, fraction_across_classes=True, idx=None, plot=True, return_cm=False):
        Performs cross-validation and returns the accuracy or confusion matrix.
    confusion_matrices(clf, method, n_repeats=100, test_size=0.3, p=1, fraction_across_classes=False, feature_type='fk'):
        Generates and saves confusion matrices for different training and testing modalities.
    predict_cells(train_modalities=['clem', 'photoactivation'], use_jon_priors=True):
        Predicts cell types based on the trained model and saves the predictions.
    plot_neurons(modality, output_filename="test.html"):
        Plots interactive 3D representations of neurons using the `navis` library and `plotly`.

    Notes:
    ------
    'fk' in the attribute names refers to the prefix 'florian kämpf' indicating these are the main features.
    """
    def __init__(self, path):
        """
        Initialize the class predictor with a given path.

        Parameters:
        -----------
        path : pathlib.Path
            The path where the confusion matrices will be saved.

        Attributes:
        -----------
        path : pathlib.Path
            The path where the confusion matrices will be saved.
        color_dict : dict
            A dictionary mapping cell types to their respective colors.
        path_to_save_confusion_matrices : pathlib.Path
            The path where confusion matrices will be saved.
        CELL_CLASS_RATIOS : dict
            A dictionary containing the ratios of different cell classes.
        """
        """
        :param path: The path where the confusion matrices will be saved.
        """
        self.path = path
        self.color_dict = {
            "integrator_ipsilateral": '#feb326b3',
            "integrator_contralateral": '#e84d8ab3',
            "dynamic_threshold": '#64c5ebb3',
            "motor_command": '#7f58afb3',
        }
        self.path_to_save_confusion_matrices = path / 'make_figures_FK_output' / 'all_confusion_matrices'
        self.real_cell_class_ratio_dict = {
            'dynamic_threshold': 22 / 539,
            'integrator_contralateral': 155.5 / 539,
            'integrator_ipsilateral': 155.5 / 539,
            'motor_command': 206 / 539
        }

        os.makedirs(self.path_to_save_confusion_matrices, exist_ok=True)

    def load_brs(self,base_path, which_brs='raphe', as_volume=True):
        """
        Loads brain regions from the specified base path and brain region set.

        Parameters:
        -----------
        base_path : pathlib.Path
            The base path where the brain regions are stored.
        which_brs : str, optional
            The specific brain region set to load (default is 'raphe').
        as_volume : bool, optional
            If True, loads the brain regions as volumes; otherwise, loads as neurons (default is True).

        Returns:
        --------
        None
        """
        if as_volume:
            load_type = 'volume'
        else:
            load_type = 'neuron'

        self.meshes = []

        for file in os.listdir(base_path.joinpath("zbrain_regions").joinpath(which_brs)):
            try:
                self.meshes.append(navis.read_mesh(base_path.joinpath("zbrain_regions").joinpath(which_brs).joinpath(file),
                                              units='um', output=load_type))
            except:
                pass

    def prepare_data_4_metric_calc(self, df, neg_control=True):
        """
        This function prepares the data for metric calculation by cleaning up the data and adding relevant information.

        Parameters:
        df (pandas.DataFrame): Input dataframe that requires data cleaning and formatting.

        neg_control (bool): If True, the 'neg_control' data records are kept. If False, those records are dropped from the dataframe.

        Returns:
        df (pandas.DataFrame): Returns the modified dataframe with cleaned data and additional information for further processing.

        The function does the following:
        - If 'neg_control' is set to False, it removes 'neg_control' records from the dataframe.
        - It drops duplicate 'cell_name' records, keeping the first occurrence.
        - If kmeans classes exist, it reads cell metadata and adds additional attributes such as 'kmeans_function', 'reliability',
          'direction_selectivity' and 'time_constant' to the dataframe. This is done separately for different imaging modalities.
        - If a new neurotransmitter information is specified, then that information is also added to the dataframe.

        """
        if not neg_control:
            df = df.loc[df['function'] != 'neg_control']

        df = df.drop_duplicates(keep='first', inplace=False, subset='cell_name')
        df = df.reset_index(drop=True)

        if self.kmeans_classes:
            for i, cell in df.iterrows():
                if cell.imaging_modality == "photoactivation":
                    temp_path_pa = self.path / 'paGFP' / cell.cell_name / f"{cell.cell_name}_metadata_with_regressor.txt"
                    with open(temp_path_pa, 'r') as f:
                        t = f.read()
                        df.loc[i, 'kmeans_function'] = t.split('\n')[11].split(' ')[2].strip('"')
                        df.loc[i, 'reliability'] = float(t.split('\n')[12].split(' ')[2].strip('"'))
                        df.loc[i, 'direction_selectivity'] = float(t.split('\n')[13].split(' ')[2].strip('"'))
                        df.loc[i, 'time_constant'] = int(t.split('\n')[14].split(' ')[2].strip('"'))

                elif cell.imaging_modality == "clem":
                    if cell.function == "neg_control":
                        df.loc[i, 'kmeans_function'] = 'neg_control'
                    elif cell.function == "to_predict":
                        df.loc[i, 'kmeans_function'] = 'to_predict'
                    else:
                        temp_path_clem = (str(cell.metadata_path)[:-4] + "_with_regressor.txt")
                        with open(temp_path_clem, 'r') as f:
                            t = f.read()
                            df.loc[i, 'kmeans_function'] = t.split('\n')[15].split(' ')[2].strip('"')
                            df.loc[i, 'reliability'] = float(t.split('\n')[16].split(' ')[2].strip('"'))
                            df.loc[i, 'direction_selectivity'] = float(t.split('\n')[17].split(' ')[2].strip('"'))
                            df.loc[i, 'time_constant'] = int(t.split('\n')[18].split(' ')[2].strip('"'))

                elif cell.imaging_modality == "EM":
                    df.loc[i, 'kmeans_function'] = df.loc[i, 'function']
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

    def calculate_metrics(self, file_name, force_new=False):
        """
        This method calculates metrics for the provided data set and saves the result to a specified file.

        Args:
            file_name(str): The name of the file where the metrics result should be saved.
            The file name should include the full path if not in the same directory.

            force_new(bool, optional): If set to True, it forces the calculation of new metrics
            even if previous metric calculations exist. By default, it is set to False which means
            it will not calculate the metrics again if they are already calculated.

        Calls:
            calculate_metric2df_seamiold: This is an inner function called for the metric calculation.

            Here, 'cells_with_to_predict' is the data for which metrics are calculated.
            The 'test.path' is used in the inner function to retrieve the test data.


        Note that the 'train_or_predict' argument in the calculation function is set to 'train',
        meaning that this function is used for training data.

        Returns:
            None. The result is saved to the specified file.
        """
        calculate_metric2df(self.cells_with_to_predict, file_name, test.path, force_new=force_new)

    def load_metrics(self, file_name, with_neg_control=False,drop_neurotransmitter=False):
        """
            This method loads metrics from a specific file and carries out several preprocessing steps on the loaded data.

            Args:
                file_name (str): The file name from which the metrics should be loaded.
                with_neg_control (bool, optional): If True, 'neg_control' records are kept. If False, 'neg_control' records are discarded; default is False.

            Returns:
                list: A list containing the features, labels, and labels_imaging_modality arrays.
                list: A list of the column labels.
                DataFrame: The preprocessed DataFrame 'all_cells'.

            Notes:
                The preprocessing steps include:
                  * Removing records where 'function' is NaN
                  * Sorting the DataFrame.
                  * Replacing certain labels in the 'function' column.
                  * Replacing NaNs with 0 in 'angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross'.
                  * Modifying 'integrator' function labels to add morphology information.
                  * Replacing certain string labels with integer indices.
                  * Extracting labels and features, and standardizing the features.

            """
        file_path = self.path / 'prediction' /'features'/ f'{file_name}_features.hdf5'

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
        all_cells.loc[:, 'function'] = all_cells['function'].str.replace(' ', '_')

        # Update 'integrator' function
        def update_integrator(df):
            integrator_mask = df['function'] == 'integrator'
            df.loc[integrator_mask, 'function'] += "_" + df.loc[integrator_mask, 'morphology']

        update_integrator(all_cells)

        # Replace strings with indices

        columns_replace_string = ['neurotransmitter', 'morphology']

        all_cells.loc[:, 'neurotransmitter'] = all_cells['neurotransmitter'].fillna('nan')



        neurotransmitter2int_dict = {'excitatory': 0, 'inhibitory': 1, 'na': 2, 'nan': 2}
        morphology2int_dict = {'contralateral': 0, 'ipsilateral': 1}

        for work_column in columns_replace_string:
            all_cells.loc[:, work_column + "_clone"] = all_cells[work_column]
            for key in eval(f'{work_column}2int_dict').keys():
                all_cells.loc[all_cells[work_column] == key, work_column] = eval(f'{work_column}2int_dict')[key]

        if drop_neurotransmitter:
            all_cells = all_cells.drop(columns='neurotransmitter')

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

    def load_cells_features(self, file, with_neg_control=False,drop_neurotransmitter=False):
        """
        This method loads cell features from a given file and filters data into categories based on their
        characteristics (such as whether they need to be predicted or whether they are negative controls)
        and imaging modality.

        Args:
            file (str): The file name from which the cell features will be loaded.
            with_neg_control (bool, optional): If True, 'neg_control' records are considered. If False, 'neg_control' records are disregarded; default is False.

        Raises:
            ValueError: If the metrics have not been loaded.

        Attributes:
            features_fk_with_to_predict, labels_fk_with_to_predict, labels_imaging_modality_with_to_predict: Loaded metrics from the file.
            labels_fk, features_fk, labels_imaging_modality, all_cells: Metric data filtered out 'to_predict' and 'neg_control' categories.
            cells: All cells with 'to_predict' attribute, present in the loaded file.
            clem_idx, pa_idx, em_idx, clem_idx_with_to_predict, pa_idx_with_to_predict, em_idx_with_to_predict: Indicator arrays for different imaging modalities.

        Notes:
            'fk' in the attribute names refers to the prefix 'florian kämpf' indicating these are the main features.
        """
        all_metric, self.column_labels, self.all_cells_with_to_predict = self.load_metrics(file, with_neg_control,drop_neurotransmitter=drop_neurotransmitter)
        self.features_fk_with_to_predict, self.labels_fk_with_to_predict, self.labels_imaging_modality_with_to_predict = all_metric
        self.labels_fk = self.labels_fk_with_to_predict[(self.labels_fk_with_to_predict != 'to_predict') & (self.labels_fk_with_to_predict != 'neg_control')]
        self.features_fk = self.features_fk_with_to_predict[(self.labels_fk_with_to_predict != 'to_predict') & (self.labels_fk_with_to_predict != 'neg_control')]
        self.labels_imaging_modality = self.labels_imaging_modality_with_to_predict[(self.labels_fk_with_to_predict != 'to_predict') & (self.labels_fk_with_to_predict != 'neg_control')]
        self.all_cells = self.all_cells_with_to_predict[(self.labels_fk_with_to_predict != 'to_predict') & (self.labels_fk_with_to_predict != 'neg_control')]

        # only select cells
        if hasattr(self, 'all_cells_with_to_predict'):

            self.cells = self.cells.set_index('cell_name').loc[self.all_cells['cell_name']].reset_index()
        else:
            raise ValueError("Metrics have not been loaded.")

        self.clem_idx = (self.cells['imaging_modality'] == 'clem').to_numpy()
        self.pa_idx = (self.cells['imaging_modality'] == 'photoactivation').to_numpy()
        self.em_idx = (self.cells['imaging_modality'] == 'EM').to_numpy()

        self.clem_idx_with_to_predict = (self.cells_with_to_predict['imaging_modality'] == 'clem').to_numpy()
        self.pa_idx_with_to_predict = (self.cells_with_to_predict['imaging_modality'] == 'photoactivation').to_numpy()
        self.em_idx_with_to_predict = (self.cells_with_to_predict['imaging_modality'] == 'EM').to_numpy()

    def load_cells_df(self, kmeans_classes=True, new_neurotransmitter=True, modalities=['pa', 'clem'], neg_control=True):
        """
        This method loads the cells dataframe and applies the preprocessing pipeline to the data.

        Args:
            kmeans_classes (bool, optional): If True, k-means classes will be considered in the data loading pipeline. Default is True.
            new_neurotransmitter (bool, optional): If True, the data loading pipeline will consider new neurotransmitter values. Default is True.
            modalities (list, optional): A list of modalities to be considered. Default is ['pa', 'clem'].
            neg_control (bool, optional): If True, will consider 'neg_control' records. If False, 'neg_control' records are disregarded; default is True.

        Attributes:
            kmeans_classes, new_neurotransmitter, modalities: Input arguments stored as class variables.
            cells_with_to_predict: Full cells dataframe after the preprocessing pipeline.
            cells: Cells dataframe with 'to_predict' and 'neg_control' records removed.

        Notes:
            The preprocessing pipeline includes the following steps:
            * Loading cells using the predictor pipeline.
            * Preparing the data for metric calculation.
            * Removing records containing 'axon' in cell names.
            * Resampling neurons to 1 micron.
            * Replacing the class label formatting in the 'class' column.
        """
        self.kmeans_classes = kmeans_classes
        self.new_neurotransmitter = new_neurotransmitter
        self.modalities = modalities

        self.cells_with_to_predict = load_cells_predictor_pipeline(path_to_data=Path(self.path),
                                                                   modalities=modalities,
                                                                   load_repaired=True)

        self.cells_with_to_predict = self.prepare_data_4_metric_calc(self.cells_with_to_predict, neg_control=neg_control)
        self.cells_with_to_predict = self.cells_with_to_predict.loc[self.cells_with_to_predict.cell_name.apply(lambda x: False if 'axon' in x else True), :]
        # resample neurons 1 micron
        self.cells_with_to_predict['swc'] = self.cells_with_to_predict['swc'].apply(lambda x: x.resample("1 micron"))
        self.cells_with_to_predict['class'] = self.cells_with_to_predict.loc[:, ['function', 'morphology']].apply(
            lambda x: x['function'].replace(" ", "_") + "_" + x['morphology'] if x['function'] == 'integrator' else x['function'].replace(" ", "_"), axis=1)

        self.cells = self.cells_with_to_predict.loc[(self.cells_with_to_predict['function'] != 'to_predict') & (self.cells_with_to_predict['function'] != 'neg_control'), :]

    def calculate_published_metrics(self):
        """
        This method calculates some published metrics over the cell data.

        Attributes:
            features_pv: Persistence vectors using the 'navis' library. The calculation uses 300 samples. A referenced article for this computation can be found at: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184
            features_ps: Similar to 'features_pv', but this includes all the computed persistence vectors.
            features_ff: Form factor of the neurons computed using the 'navis' library.

        Notes:
            The 'navis.form_factor' function is parallelized to 15 cores and uses 300 samples for its computation. The method used is described in the article: https://link.springer.com/article/10.1007/s12021-017-9341-1
        """
        self.features_pv = np.stack([navis.persistence_vectors(x, samples=300)[0] for x in self.cells.swc])[:, 0,
                           :]  # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184
        self.features_ps = np.stack([navis.persistence_vectors(x, samples=300)[1] for x in
                                     self.cells.swc])  # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0182184
        self.features_ff = navis.form_factor(navis.NeuronList(self.cells.swc), n_cores=15, parallel=True,
                                             num=300)  # https://link.springer.com/article/10.1007/s12021-017-9341-1

    def select_features(self, train_mod: str, test_mod: str, plot=False, use_assessment_per_class=False, which_selection='SKB', use_std_scale=False):
        """
        Selects the optimal features for a given training and testing modality.

        Parameters:
        -----------
        train_mod : str
            The training modality.
        test_mod : str
            The testing modality.
        plot : bool, optional
            If True, plots the feature selection results. Default is False.
        use_assessment_per_class : bool, optional
            If True, uses assessment per class for feature selection. Default is False.
        which_selection : str, optional
            The feature selection method to use ('SKB', 'PI', or custom scorer). Default is 'SKB'.
        use_std_scale : bool, optional
            If True, uses standard scaling for penalty calculation. Default is False.

        Returns:
        --------
        bool_features_2_use : numpy.ndarray
            Boolean array indicating the selected features.
        max_accuracy_key : str
            The key corresponding to the maximum accuracy.
        train_mod : str
            The training modality.
        test_mod : str
            The testing modality.
        """
        def calc_penalty(temp_list):
            p = 2  # You can adjust this power as needed
            penalty = np.mean(temp_list) / np.exp(np.std(temp_list) ** p)
            return penalty

        def find_optimum_SKB(features_train, labels_train, features_test, labels_test, train_test_identical,
                             train_contains_test, train_mod, use_std_scale=False):
            """
            Finds the optimal number of features using SelectKBest for feature selection.

            Parameters:
            -----------
            features_train : numpy.ndarray
                Training feature set.
            labels_train : numpy.ndarray
                Training labels.
            features_test : numpy.ndarray
                Test feature set.
            labels_test : numpy.ndarray
                Test labels.
            train_test_identical : bool
                Indicates if the training and test sets are identical.
            train_contains_test : bool
                Indicates if the training set contains the test set.
            train_mod : str
                The training modality.
            use_std_scale : bool, optional
                If True, uses standard scaling for penalty calculation. Default is False.

            Returns:
            --------
            pred_correct_dict_over_n : dict
                Dictionary containing prediction correctness over different numbers of features.
            pred_correct_dict_over_n_per_class : dict
                Dictionary containing prediction correctness per class over different numbers of features.
            used_features_idx_over_n : dict
                Dictionary containing the indices of used features over different numbers of features.
            """


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
            """
            Finds the optimal number of features using permutation importance for feature selection.

            Parameters:
            -----------
            features_train : numpy.ndarray
                Training feature set.
            labels_train : numpy.ndarray
                Training labels.
            features_test : numpy.ndarray
                Test feature set.
            labels_test : numpy.ndarray
                Test labels.
            train_test_identical : bool
                Indicates if the training and test sets are identical.
            train_contains_test : bool
                Indicates if the training set contains the test set.
            train_mod : str
                The training modality.
            use_std_scale : bool, optional
                If True, uses standard scaling for penalty calculation. Default is False.

            Returns:
            --------
            pred_correct_dict_over_n : dict
                Dictionary containing prediction correctness over different numbers of features.
            pred_correct_dict_over_n_per_class : dict
                Dictionary containing prediction correctness per class over different numbers of features.
            used_features_idx_over_n : dict
                Dictionary containing the indices of used features over different numbers of features.
            """
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
            """
            Finds the optimal number of features using a custom scorer for feature selection.

            Parameters:
            -----------
            custom_scorer : object
                The custom scoring model used for feature selection.
            features_train : numpy.ndarray
                Training feature set.
            labels_train : numpy.ndarray
                Training labels.
            features_test : numpy.ndarray
                Test feature set.
            labels_test : numpy.ndarray
                Test labels.
            train_test_identical : bool
                Indicates if the training and test sets are identical.
            train_contains_test : bool
                Indicates if the training set contains the test set.
            train_mod : str
                The training modality.
            use_std_scale : bool, optional
                If True, uses standard scaling for penalty calculation. Default is False.

            Returns:
            --------
            pred_correct_dict_over_n : dict
                Dictionary containing prediction correctness over different numbers of features.
            pred_correct_dict_over_n_per_class : dict
                Dictionary containing prediction correctness per class over different numbers of features.
            used_features_idx_over_n : dict
                Dictionary containing the indices of used features over different numbers of features.
            """
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

        mod2idx = {'all': np.full(len(self.pa_idx), True), 'pa': self.pa_idx, 'clem': self.clem_idx}

        features_train = self.features_fk[mod2idx[train_mod]]
        labels_train = self.labels_fk[mod2idx[train_mod]]
        features_test = self.features_fk[mod2idx[test_mod]]
        labels_test = self.labels_fk[mod2idx[test_mod]]
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

        return bool_features_2_use, max_accuracy_key, train_mod, test_mod

    def select_features_RFE(self, train_mod, test_mod, cv=False, estimator=None, scoring=None, cv_method=None, save_features=False,cv_method_RFE='lpo'):
        mod2idx = {'all': np.full(len(self.pa_idx), True), 'pa': self.pa_idx, 'clem': self.clem_idx}
        """
        Selects features using Recursive Feature Elimination (RFE) or RFECV.

        Args:
            train_mod (str): The training modality.
            test_mod (str): The testing modality.
            cv (bool, optional): If True, uses cross-validation for feature selection. Default is False.
            estimator (object, optional): The estimator to use for feature selection. Default is None.
            scoring (str or list, optional): The scoring method(s) to use. Default is None.
            cv_method (object or list, optional): The cross-validation method(s) to use. Default is None.
            save_features (bool, optional): If True, saves the selected features. Default is False.

        Returns:
            None
        """
        if estimator is None:
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
        else:
            if type(scoring) != list:
                all_estimator = [estimator]
            else:
                all_estimator = estimator

        if scoring is None:
            all_scoring = ['accuracy', 'balanced_accuracy', 'average_precision', 'f1', 'f1_weighted']
        else:
            if type(scoring) != list:
                all_scoring = [scoring]
            else:
                all_scoring = scoring

        if cv_method is None:
            cv_method = [ShuffleSplit(n_splits=100, test_size=0.3, random_state=0), LeavePOut(p=1), StratifiedKFold(n_splits=5), KFold(n_splits=5)]
        else:
            if type(cv_method) != list:
                cv_method = [cv_method]

        if not cv:
            for estimator in all_estimator:
                acc_list = []
                for i in np.arange(1, self.features_fk.shape[1] + 1):
                    selector = RFE(estimator, n_features_to_select=i, step=1).fit(self.features_fk, self.labels_fk)
                    acc = self.do_cv(method=cv_method_RFE, clf=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), feature_type='fk',
                                     train_mod=train_mod, test_mod=test_mod, figure_label=str(estimator) + "_n" + str(i), fraction_across_classes=True,n_repeats=100, idx=selector.support_, plot=False)
                    acc_list.append(acc)
                plt.figure()
                plt.plot(acc_list)
                plt.axvline(np.nanargmax(acc_list), c='red', alpha=0.3)
                plt.text(np.nanargmax(acc_list) + 1, np.mean(plt.ylim()), f'n = {np.nanargmax(acc_list) + 1}', fontsize=12, color='red', ha='left', va='bottom')
                plt.gca().set_xticks(np.arange(0, self.features_fk.shape[1], 3), np.arange(1, self.features_fk.shape[1] + 2, 3))
                plt.title(f"{str(estimator)}\nAccuracy {acc_list[np.nanargmax(acc_list)]}%", fontsize='small')
                selector = RFE(estimator, n_features_to_select=np.nanargmax(acc_list) + 1, step=1).fit(self.features_fk, self.labels_fk)

                temp_str = f"Estimator_{str(estimator)}\nfeatures_{np.sum(selector.support_)}"
                temp_str = '\n'.join([x.split('(')[0] for x in temp_str.split('\n')])
                print(temp_str, '\n')

                temp_path = self.path / 'prediction' / 'RFE'
                os.makedirs(temp_path, exist_ok=True)

                le = [Patch(facecolor='white', edgecolor='white', label=x) for x in np.array(self.column_labels)[selector.support_]]
                plt.legend(handles=le, frameon=False, fontsize=6, loc=[1, 0.0], bbox_to_anchor=(1, 0))
                plt.subplots_adjust(left=0.1, right=0.65, top=0.80, bottom=0.1)

                temp_str = str(int(np.round(np.nanmax(acc_list)))) + "_" + temp_str.replace('\n', "_")

                plt.savefig(temp_path / f"{temp_str}.png")

                plt.show()
                selector = RFE(estimator, n_features_to_select=np.argmax(acc_list) + 1, step=1).fit(self.features_fk, self.labels_fk)
                self.select_train_mod = train_mod
                self.select_test_mod =  test_mod
                self.select_method=str(estimator)
                self.confusion_matrices(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), method='lpo',idx=selector.support_)
            if save_features:
                selector = RFE(estimator, n_features_to_select=np.argmax(acc_list) + 1, step=1).fit(self.features_fk, self.labels_fk)

                self.reduced_features_idx = selector.support_
                self.select_train_mod = train_mod
                self.select_test_mod = test_mod
                self.select_method = str(str(estimator))


        elif cv:
            for estimator in all_estimator:
                for scoring in all_scoring:
                    for cv in cv_method:
                        selector = RFECV(estimator, step=1, cv=cv).fit(self.features_fk, self.labels_fk)
                        temp_str = f"Estimator_{str(estimator)}\nScoring_{str(scoring)}\nCV_{str(cv)}\nfeatures_{np.sum(selector.support_)}"
                        temp_str = '\n'.join([x.split('(')[0] for x in temp_str.split('\n')])
                        print(temp_str, '\n')
                        accuracy = self.do_cv(method='lpo', clf=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), feature_type='fk',
                                              train_mod=train_mod, test_mod=test_mod, figure_label=temp_str,
                                              fraction_across_classes=False, idx=selector.support_, plot=True)
                        temp_path = self.path / 'prediction' / 'RFECV'
                        os.makedirs(temp_path, exist_ok=True)
                        temp_str = str(int(np.round(accuracy))) + "_" + temp_str.replace('\n', "_")

                        plt.savefig(temp_path / f"{temp_str}.png")
                        pass
            if save_features:
                self.reduced_features_idx = selector.support_
                self.select_train_mod = train_mod
                self.select_test_mod = test_mod
                self.select_method = str(str(estimator) + "_" + str(all_scoring) + "_" + str(cv_method))

    def do_cv(self, method: str, clf, feature_type, train_mod, test_mod, n_repeats=100, test_size=0.3, p=1, ax=None, figure_label='error:no figure label', spines_red=False,
              fraction_across_classes=True, idx=None, plot=True, return_cm=False):
        """
        Perform cross-validation on the given classifier and dataset.

        Parameters:
        -----------
        method : str
            The cross-validation method to use ('lpo' for LeavePOut or 'ss' for ShuffleSplit).
        clf : object
            The classifier to use for training and prediction.
        feature_type : str
            The type of features to use ('fk', 'pv', 'ps', or 'ff').
        train_mod : str
            The training modality.
        test_mod : str
            The testing modality.
        n_repeats : int, optional
            The number of repeats for ShuffleSplit. Default is 100.
        test_size : float, optional
            The test size for ShuffleSplit. Default is 0.3.
        p : int, optional
            The number of samples to leave out for LeavePOut. Default is 1.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the confusion matrix. Default is None.
        figure_label : str, optional
            The label for the figure. Default is 'error:no figure label'.
        spines_red : bool, optional
            If True, set the spines of the plot to red. Default is False.
        fraction_across_classes : bool, optional
            If True, normalize the confusion matrix by true labels. Default is True.
        idx : array-like, optional
            The indices of the features to use. Default is None.
        plot : bool, optional
            If True, plot the confusion matrix. Default is True.
        return_cm : bool, optional
            If True, return the confusion matrix. Default is False.

        Returns:
        --------
        float or numpy.ndarray
            The accuracy score if return_cm is False, otherwise the confusion matrix.
        """
        def check_duplicates(train, test):
            """
            Check for duplicate rows between the training and test datasets.

            Args:
                train (numpy.ndarray): The training dataset.
                test (numpy.ndarray): The test dataset.

            Returns:
                bool: True if duplicate rows are found, False otherwise.

            Notes:
                This function converts both datasets to sets of rows and finds common rows between them.
                If duplicates are found, it prints the number of duplicate rows and returns True.
                Otherwise, it returns False.
            """
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
                return self.features_fk[model_to_index[mode]][:, idx]
            elif feature_type == 'pv':
                return self.features_pv[model_to_index[mode]]
            elif feature_type == 'ps':
                return self.features_ps[model_to_index[mode]]
            elif feature_type == 'ff':
                return self.features_ff[model_to_index[mode]]

            labels_train = self.labels_fk[mod2idx[train_mod]]
            labels_test = self.labels_fk_[mod2idx[test_mod]]

        if idx is None:
            idx = self.reduced_features_idx
        features_train = extract_features(feature_type, mod2idx, train_mod, idx)
        features_test = extract_features(feature_type, mod2idx, test_mod, idx)
        labels_train = self.labels_fk[mod2idx[train_mod]]
        labels_test = self.labels_fk[mod2idx[test_mod]]

        if test_mod == train_mod:
            check_test_equals_train = True
        else:
            check_test_equals_train = False
        if test_mod == 'all' and train_mod != 'all':
            check_train_in_test = True
            check_test_in_train = False
        elif train_mod == 'all' and test_mod != 'all':
            check_train_in_test = False
            check_test_in_train = True
        else:
            check_train_in_test = False
            check_test_in_train = False

        scaler = StandardScaler()
        features_train = scaler.fit_transform(features_train)
        features_test = scaler.transform(features_test)

        true_labels = []
        pred_labels = []

        if method == 'lpo':
            splitter = LeavePOut(p=p)
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
                bool_train = np.full_like(mod2idx[test_mod], False)
                bool_test = np.full_like(mod2idx[test_mod], False)
                bool_train[train_index] = True
                bool_test[test_index] = True

                clf_work = clone(clf)
                X_train, X_test, y_train, y_test = (features_train[bool_train * mod2idx[train_mod]],
                                                    features_train[bool_test * mod2idx[test_mod]],
                                                    labels_train[bool_train * mod2idx[train_mod]],
                                                    labels_train[bool_test * mod2idx[test_mod]])
                clf_work.fit(X_train, y_train)
                if y_test.size != 0:

                    try:
                        pred_labels.extend(clf_work.predict(X_test))
                        true_labels.extend(list(y_test))
                    except:
                        pass
        elif check_train_in_test:
            true_labels = []
            pred_labels = []
            for train_index, test_index in splitter.split(features_test):
                bool_train = np.full_like(mod2idx[test_mod], False)
                bool_test = np.full_like(mod2idx[test_mod], False)
                bool_train[train_index] = True
                bool_test[test_index] = True

                clf_work = clone(clf)
                X_train, X_test, y_train, y_test = (features_test[bool_train * mod2idx[train_mod]],
                                                    features_test[bool_test * mod2idx[test_mod]],
                                                    labels_test[bool_train * mod2idx[train_mod]],
                                                    labels_test[bool_test * mod2idx[test_mod]])
                clf_work.fit(X_train, y_train)
                if y_test.size != 0:

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
                ss_train = copy.deepcopy(splitter)
                ss_test = copy.deepcopy(splitter)
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
            split = f"{(1 - test_size) * 100}:{test_size * 100}"
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
                    ax.set_title(f"Confusion Matrix (SS {split} x{n_repeats})" + f'\nAccuracy: {round(accuracy_score(true_labels, pred_labels) * 100, 2)}%' + f'\n{figure_label}')
                elif method == 'lpo':
                    ax.set_title(f"Confusion Matrix (LPO = {p})" + f'\nAccuracy: {round(accuracy_score(true_labels, pred_labels) * 100, 2)}%' + f'\n{figure_label}')
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
        if return_cm:
            return cm
        else:
            return round(accuracy_score(true_labels, pred_labels) * 100, 2)

    def confusion_matrices(self, clf, method: str, n_repeats=100,
                           test_size=0.3, p=1,
                           fraction_across_classes=False, feature_type='fk',idx=None):
        """
        Generate and save confusion matrices for different training and testing modalities.

        Parameters:
        -----------
        clf : object
            The classifier to use for training and prediction.
        method : str
            The cross-validation method to use ('lpo' for LeavePOut or 'ss' for ShuffleSplit).
        n_repeats : int, optional
            The number of repeats for ShuffleSplit. Default is 100.
        test_size : float, optional
            The test size for ShuffleSplit. Default is 0.3.
        p : int, optional
            The number of samples to leave out for LeavePOut. Default is 1.
        fraction_across_classes : bool, optional
            If True, normalize the confusion matrix by true labels. Default is False.
        feature_type : str, optional
            The type of features to use ('fk', 'pv', 'ps', or 'ff'). Default is 'fk'.

        Returns:
        --------
        None
        """

        suptitle = feature_type.upper() + "_features_"
        if method == 'lpo':
            suptitle += f'lpo{p}'
        elif method == 'ss':
            suptitle += f'ss_{int((1 - test_size) * 100)}_{int((test_size) * 100)}'

        if feature_type == 'fk':
            suptitle += f'{self.select_train_mod}_{self.select_test_mod}_{self.select_method}'

        target_train_test = ['ALLCLEM', 'CLEMCLEM', 'PAPA']
        fig, ax = plt.subplots(3, 3, figsize=(20, 20))
        for train_mod, loc_x in zip(['ALL', "CLEM", "PA"], range(3)):
            for test_mod, loc_y in zip(['ALL', "CLEM", "PA"], range(3)):
                spines_red = train_mod + test_mod in target_train_test
                self.do_cv(method, clf, feature_type, train_mod.lower(), test_mod.lower(), figure_label=f'{train_mod}_{test_mod}',
                           ax=ax[loc_x, loc_y], spines_red=spines_red, fraction_across_classes=fraction_across_classes, n_repeats=n_repeats, test_size=test_size, p=p,idx=idx)
        fig.suptitle(suptitle, fontsize='xx-large')
        plt.savefig(self.path_to_save_confusion_matrices / f'{suptitle}.png')
        plt.savefig(self.path_to_save_confusion_matrices / f'{suptitle}.pdf')
        plt.show()


    def calculate_verification_metrics(self,calculate_smat=True,with_kunst=True):



        #check if ipsi got asigned as contra and vice versa
        print(self.prediction_predict_df.groupby(['morphology_clone', 'prediction']).size())

        train_cells = test.prediction_train_df
        to_predict_cells = test.prediction_predict_df[test.prediction_predict_df.function == 'to_predict']
        neg_control_cells = test.prediction_predict_df[test.prediction_predict_df.function == 'neg_control']

        names_dt = train_cells.loc[(train_cells['function'] == 'dynamic_threshold'), 'cell_name']
        names_ii = train_cells.loc[(train_cells['function'] == 'integrator_ipsilateral'), 'cell_name']
        names_ci = train_cells.loc[(train_cells['function'] == 'integrator_contralateral'), 'cell_name']
        names_mc = train_cells.loc[(train_cells['function'] == 'motor_command'), 'cell_name']
        names_nc = neg_control_cells['cell_name']
        if with_kunst:
            path = test.path / 'prediction' / f'smat_fish_with_kunst.pkl'
        else:
            path = test.path / 'prediction' / f'smat_fish.pkl'
        if calculate_smat:
            self.smat_fish = calculate_zebrafish_nblast_matrix(self.cells_with_to_predict,path_to_data=self.path,with_kunst=with_kunst, return_smat_obj=True, prune=False)
            with open(path,'wb') as f:
                pickle.dump(self.smat_fish, f)
        else:
            with open(path,'rb') as f:
                self.smat_fish = pickle.load(f)
        self.nb_train = nblast_two_groups_custom_matrix(train_cells, train_cells, custom_matrix=self.smat_fish, shift_neurons=False)
        self.per_class = self.nb_train.groupby([self.nb_train.index]).mean().T.groupby(self.nb_train.index).mean()
        self.nb_train.index =  [train_cells.loc[train_cells.cell_name == x, 'function'].iloc[0] for x in self.nb_train.index]
        self.nb_train.columns = [train_cells.loc[train_cells.cell_name == x, 'function'].iloc[0] for x in self.nb_train.columns]




        self.nb_train_nc = nblast_two_groups_custom_matrix(train_cells, neg_control_cells, custom_matrix=self.smat_fish, shift_neurons=False)
        self.nb_train_predict = nblast_two_groups_custom_matrix(train_cells, to_predict_cells, custom_matrix=self.smat_fish, shift_neurons=False)

        self.nb_matches_cells_train = navis.nbl.extract_matches(self.nb_train, 2)
        self.nb_matches_cells_nc = navis.nbl.extract_matches(self.nb_train_nc.T, 2)
        self.nb_matches_cells_predict = navis.nbl.extract_matches(self.nb_train_predict.T, 2)

        self.nblast_values_dt = navis.nbl.extract_matches(self.nb_train.loc[names_dt, names_dt], 2)
        self.nblast_values_ii = navis.nbl.extract_matches(self.nb_train.loc[names_ii, names_ii], 2)
        self.nblast_values_ci = navis.nbl.extract_matches(self.nb_train.loc[names_ci, names_ci], 2)
        self.nblast_values_mc = navis.nbl.extract_matches(self.nb_train.loc[names_mc, names_mc], 2)

        z_score_dt = lambda x: abs((x - np.mean(list(self.nblast_values_dt.score_2))) / np.std(list(self.nblast_values_dt.score_2)))
        z_score_ii = lambda x: abs((x - np.mean(list(self.nblast_values_ii.score_2))) / np.std(list(self.nblast_values_ii.score_2)))
        z_score_ci = lambda x: abs((x - np.mean(list(self.nblast_values_ci.score_2))) / np.std(list(self.nblast_values_ci.score_2)))
        z_score_mc = lambda x: abs((x - np.mean(list(self.nblast_values_mc.score_2))) / np.std(list(self.nblast_values_mc.score_2)))

        cutoff = self.nb_matches_cells_train.loc[:, 'score_2'].quantile(.1)
        print(f'{(self.nb_matches_cells_nc["score_1"] >= cutoff).sum()} of {self.nb_matches_cells_nc.shape[0]} neg_control cells pass NBlast general test.')

        subset_predict_cells = list(self.nb_matches_cells_predict.loc[self.nb_matches_cells_predict['score_1'] >= cutoff, 'id'])

        OCSVM = OneClassSVM(gamma='scale', kernel='poly').fit(test.prediction_train_features)
        IF = IsolationForest(contamination=0.1, random_state=42).fit(test.prediction_train_features)
        LOF = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(test.prediction_train_features)


        self.prediction_predict_df.loc[:, 'OCSVM'] = OCSVM.predict(test.prediction_predict_features) == 1
        self.prediction_predict_df.loc[:, 'IF'] = IF.predict(test.prediction_predict_features) == 1
        self.prediction_predict_df.loc[:, 'LOF'] = LOF.predict(test.prediction_predict_features) == 1


    def predict_cells(self, train_modalities=['clem', 'photoactivation'], use_jon_priors=True,suffix=''):
        """
        Predicts cell types based on selected training modalities and optionally using Jon's priors.

        This function performs the following steps:
        1. Selects training modalities and aligns dataframes.
        2. Selects training data based on the specified modalities.
        3. Excludes certain cells based on various criteria.
        4. Trains a Linear Discriminant Analysis (LDA) classifier.
        5. Predicts cell types and probabilities for the test data.
        6. Scales the predicted probabilities.
        7. Exports the predictions to Excel files.
        8. Updates metadata files with predictions.

        Parameters:
        -----------
        train_modalities : list of str, optional
            List of training modalities to use for prediction. Default is ['clem', 'photoactivation'].
        use_jon_priors : bool, optional
            If True, uses Jon's priors for prediction. Default is True.

        Returns:
        --------
        None
        """

        if use_jon_priors:
            suffix = suffix + "_jon_prior"
        if suffix != '' and suffix[0]!= "_":
            suffix = "_" + suffix


        # modality train selection
        modality2idx = {'clem': self.clem_idx,
                        'photoactivation': self.pa_idx}
        selected_indices = None
        for idx in train_modalities:
            if selected_indices is None:
                selected_indices = modality2idx[idx]
            else:
                selected_indices = selected_indices + modality2idx[idx]

        # align the two dfs
        super_df = pd.merge(self.all_cells_with_to_predict, self.cells_with_to_predict.loc[:, ['cell_name', 'metadata_path', 'comment','swc']], on=['cell_name'], how='inner')

        # Select data based on train_modalities
        self.prediction_train_df = pd.merge(self.all_cells, self.cells.loc[:, ['cell_name', 'metadata_path', 'comment','swc']], on=['cell_name'], how='inner')
        self.prediction_train_df = self.prediction_train_df[self.prediction_train_df.imaging_modality.isin(train_modalities)]
        self.prediction_train_features = self.features_fk[selected_indices][:, self.reduced_features_idx]
        self.prediction_train_labels = self.labels_fk[selected_indices]

        exclude_axon_bool = (self.all_cells_with_to_predict.cell_name.apply(lambda x: False if 'axon' in x else True)).to_numpy()
        exclude_train_bool = (~self.all_cells_with_to_predict.cell_name.isin(self.prediction_train_df.cell_name)).to_numpy()
        exclude_not_to_predict = ((self.all_cells_with_to_predict.function == 'to_predict')|(self.all_cells_with_to_predict.function == 'neg_control')).to_numpy()
        exclude_nan = np.any(~np.isnan(self.features_fk_with_to_predict), axis=1)
        exclude_reticulospinal = np.array([('reticulospinal' not in str(x)) for x in super_df.comment])
        exclude_myelinated = np.array([('myelinated' not in str(x)) for x in super_df.comment])
        exclude_bool = exclude_train_bool * exclude_axon_bool * exclude_not_to_predict * exclude_nan * exclude_reticulospinal * exclude_myelinated

        self.prediction_predict_df = super_df.loc[exclude_bool, :]
        self.prediction_predict_features = self.features_fk_with_to_predict[exclude_bool][:, self.reduced_features_idx]
        self.prediction_predict_labels = self.labels_fk_with_to_predict[exclude_bool]
        if use_jon_priors:
            priors = [self.real_cell_class_ratio_dict[x] for x in np.unique(self.prediction_train_labels)]  # Hindbrain,Rhombomere 1-3’: {INTs: 311, MCs: 206, DTs: 22}),
        else:
            priors = [len(self.prediction_train_labels[self.prediction_train_labels == x]) / len(self.prediction_train_labels) for x in np.unique(self.prediction_train_labels)]
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=priors)
        clf.fit(self.prediction_train_features, self.prediction_train_labels.flatten())

        (self.prediction_train_df.function == self.prediction_train_labels).all()
        self.prediction_predict_df.loc[:, ['DT_proba', 'CI_proba', 'II_proba', 'MC_proba']] = clf.predict_proba(self.prediction_predict_features)
        predicted_int_temp = np.argmax(self.prediction_predict_df.loc[:, ['DT_proba', 'CI_proba', 'II_proba', 'MC_proba']].to_numpy(), axis=1)
        self.prediction_predict_df['prediction'] = [clf.classes_[x] for x in predicted_int_temp]

        cm = self.do_cv(method='lpo',
                        clf=LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
                        feature_type='fk',
                        train_mod='all',
                        test_mod='clem',
                        fraction_across_classes=False,
                        n_repeats=100,
                        test_size=0.3,
                        p=1,
                        return_cm=True)
        true_positive_dict = {}
        for i, k in enumerate(clf.classes_):
            true_positive_dict[k] = cm[i, i]
        true_positive_flat = list(true_positive_dict.values())

        self.prediction_predict_df.loc[:, ['DT_proba_scaled', 'CI_proba_scaled', 'II_proba_scaled', 'MC_proba_scaled']] = clf.predict_proba(self.prediction_predict_features) * true_positive_flat
        predicted_int_temp = np.argmax(self.prediction_predict_df.loc[:, ['DT_proba_scaled', 'CI_proba_scaled', 'II_proba_scaled', 'MC_proba_scaled']].to_numpy(), axis=1)
        self.prediction_predict_df['prediction_scaled'] = [clf.classes_[x] for x in predicted_int_temp]
        export_columns = ['cell_name', 'morphology_clone', 'neurotransmitter_clone', 'prediction', 'DT_proba', 'CI_proba', 'II_proba',
                          'MC_proba', 'prediction_scaled', 'DT_proba_scaled', 'CI_proba_scaled', 'II_proba_scaled',
                          'MC_proba_scaled']
        self.prediction_predict_df.loc[self.prediction_predict_df['imaging_modality'] == 'clem', export_columns].to_excel(self.path / 'clem_zfish1' / f'clem_cell_prediction{suffix}.xlsx')
        self.prediction_predict_df.loc[self.prediction_predict_df['imaging_modality'] == 'EM', export_columns].to_excel(self.path / 'em_zfish1' / f'em_cell_prediction{suffix}.xlsx')

        for i, item in self.prediction_predict_df.iterrows():
            with open(item["metadata_path"], 'r') as f:
                t = f.read()
            t = t.replace('\n[others]\n', '')
            prediction_str = f"Prediction: {item['prediction']}\n"
            prediction_scaled_str = f"Prediction_scaled: {item['prediction_scaled']}\n"
            proba_str = (f"Proba_prediction: "
                         f"DT: {round(item['DT_proba'], 2)} "
                         f"CI: {round(item['CI_proba'], 2)} "
                         f"II: {round(item['II_proba'], 2)} "
                         f"MC: {round(item['MC_proba'], 2)}\n")
            proba_scaled_str = (f"Proba_prediction_scaled: "
                                f"DT: {round(item['DT_proba_scaled'], 2)} "
                                f"CI: {round(item['CI_proba_scaled'], 2)} "
                                f"II: {round(item['II_proba_scaled'], 2)} "
                                f"MC: {round(item['MC_proba_scaled'], 2)}")

            if not t[-1:] == '\n':
                t = t + '\n'

            new_t = (t + prediction_str + proba_str + prediction_scaled_str + proba_scaled_str)
            output_path = Path(str(item['metadata_path'])[:-4] + f"_with_prediction{suffix}.txt")

            if output_path.exists():
                os.remove(output_path)

            with open(output_path, 'w+') as f:
                f.write(new_t)

    def plot_neurons(self, modality: str,scaled:bool=True, output_filename: str = "test.html") -> None:
        """
        Plots interactive 3D representations of neurons using the `navis` library and `plotly`.

        Parameters:
        - modality (str): The imaging modality to filter the neurons by (e.g., 'clem', 'photoactivation').
        - output_filename (str): The filename for the output HTML file containing the plot. Default is "test.html".

        This function:
        1. Loads brain structures using the `load_brs` function.
        2. Filters the neurons based on the specified imaging modality.
        3. Sets the color for each neuron based on its predicted function.
        4. Adjusts the radius of the neuron nodes for better visualization.
        5. Plots the neurons in 3D using `navis` and `plotly`.
        6. Saves the plot to an HTML file.

        Returns:
        None
        """
        os.makedirs(self.path / 'prediction' / 'interactive_predictions' / modality,exist_ok=True)
        output_filename = self.path / 'prediction' / 'interactive_predictions' / modality / output_filename
        if scaled:
            scaled_suffix = "_scaled"
        else:
            scaled_suffix = ""

        try:
            self.load_brs(self.path)
        except Exception as e:
            raise RuntimeError(f"Failed to load brain structures: {str(e)}")

        valid_modalities = self.prediction_predict_df['imaging_modality'].unique()
        if modality not in valid_modalities:
            raise ValueError(f"Invalid modality '{modality}'. Must be one of: {valid_modalities}")

        # Filter neurons by modality
        sub_df = self.prediction_predict_df[self.prediction_predict_df['imaging_modality'] == modality].copy().sort_values(f'prediction{scaled_suffix}')
        colors = [self.color_dict[pred] for pred in sub_df[f'prediction{scaled_suffix}']]

        # Prepare neurons for visualization
        def prepare_neuron(row):
            row['swc'].nodes['radius'] = 0.5
            row['swc'].nodes.loc[row.swc.nodes.node_id == row.swc.nodes.node_id.min(), 'radius'] = 2
            row['swc'].soma = 0
            return row

        sub_df = sub_df.apply(prepare_neuron, axis=1)

        # Create 3D plot
        plot_params = {'backend': 'plotly', 'width': 1920, 'height': 1080, 'hover_name': True}
        fig = navis.plot3d(self.meshes, **plot_params)
        fig = navis.plot3d(navis.NeuronList(sub_df.swc), fig=fig, colors=colors, **plot_params)

        # Configure plot layout
        fig.update_layout(
            scene={
                'xaxis': {'autorange': 'reversed'},
                'yaxis': {'autorange': True},
                'zaxis': {'autorange': True},
                'aspectmode': 'data',
                'aspectratio': {'x': 1, 'y': 1, 'z': 1}
            }
        )

        plotly.offline.plot(fig, filename=str(output_filename), auto_open=False, auto_play=False)

    def add_new_morphology_annotation(self):
        dt_annotation = pd.read_excel(self.path / 'prediction' / 'auxiliary_files' / 'dt_morphology_annotation_gregor.xlsx')
        morphology_in_features_dict = {'ipsilateral':np.unique(self.features_fk[self.all_cells['morphology_clone']=='ipsilateral',0])[0],
                                       'contralateral':np.unique(self.features_fk[self.all_cells['morphology_clone']=='contralateral',0])[0]}


        for i,cell in dt_annotation.iterrows():
            self.cells.loc[self.cells['cell_name'] == cell['cell_name'], "morphology"] = cell['morphology']
            self.cells_with_to_predict.loc[self.cells_with_to_predict['cell_name'] == cell['cell_name'], "morphology"] = cell['morphology']
            self.all_cells.loc[self.all_cells['cell_name'] == cell['cell_name'], "morphology_clone"] = cell['morphology']
            self.all_cells_with_to_predict.loc[self.all_cells_with_to_predict['cell_name'] == cell['cell_name'], "morphology_clone"] = cell['morphology']

        for morphology in morphology_in_features_dict.keys():
            self.features_fk[(self.all_cells['morphology_clone'] == morphology)&
                             (self.all_cells['function'] == 'dynamic_threshold')&
                             (self.all_cells['imaging_modality'] == 'clem'), 0] = morphology_in_features_dict[morphology]
            self.features_fk_with_to_predict[(self.all_cells_with_to_predict['morphology_clone'] == morphology) &
                                             (self.all_cells_with_to_predict['function'] == 'dynamic_threshold')&
                                             (self.all_cells_with_to_predict['imaging_modality'] == 'clem'), 0] = morphology_in_features_dict[morphology]

    def remove_incomplete(self,growth_cone=False,exits_volume=False,truncated=False):
        if growth_cone:
            bool_growth_cone = ['growth cone' not in str(x) for x in self.cells.comment]
        else:
            bool_growth_cone = [True for x in self.cells.comment]
        if exits_volume:
            bool_exits = ['exit' not in str(x) for x in self.cells.comment]
        else:
            bool_exits = [True for x in self.cells.comment]
        if truncated:
            bool_truncated = ['truncated' not in str(x) for x in self.cells.comment]
        else:
            bool_truncated = [True for x in self.cells.comment]
        bool_all = np.array(bool_truncated)*np.array(bool_growth_cone)*np.array(bool_exits)
        self.cells = self.cells[bool_all]
        self.all_cells = self.all_cells[bool_all]
        self.features_fk = self.features_fk[bool_all]
        self.labels_fk = self.labels_fk[bool_all]
        self.pa_idx = (self.cells['imaging_modality'] == 'photoactivation').to_numpy()
        self.clem_idx = (self.cells['imaging_modality'] == 'clem').to_numpy()

if __name__ == "__main__":
    # load metrics and cells
    test = class_predictor(Path(r'D:\hindbrain_structure_function\nextcloud'))
    test.load_cells_df(kmeans_classes=True, new_neurotransmitter=True, modalities=['pa', 'clem', 'em', 'clem_predict'], neg_control=True)
    test.calculate_metrics('FINAL_CLEM_CLEMPREDICT_EM_PA') #
    # test.calculate_published_metrics()
    test.load_cells_features('FINAL_CLEM_CLEMPREDICT_EM_PA', with_neg_control=True,drop_neurotransmitter=False)

    #throw out truncated, exits and growth cone
    test.remove_incomplete()

    #apply gregors manual morphology annotations

    test.add_new_morphology_annotation()



    # select features
    #test.select_features_RFE('all', 'clem', cv=False,cv_method_RFE='lpo') #runs through all estimator
    #test.select_features_RFE('all','clem',cv=False,save_features=True,estimator=Perceptron(random_state=0),cv_method_RFE='lpo')
    test.select_features_RFE('all', 'clem', cv=False, save_features=True, estimator=LogisticRegression(random_state=0),cv_method_RFE='lpo')



    # select classifiers for the confusion matrices
    clf_fk = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    n_estimators_rf = 100
    clf_pv = RandomForestClassifier(n_estimators=n_estimators_rf)
    clf_ps = RandomForestClassifier(n_estimators=n_estimators_rf)
    clf_ff = RandomForestClassifier(n_estimators=n_estimators_rf)

    # make confusion matrices
    test.confusion_matrices(clf_fk, method='lpo')

    #predict cells
    test.predict_cells(use_jon_priors=True,suffix='_optimize_all_predict') #optimize_all_predict means to go for the 82.05%, alternative is balance_all_pa which goes to 79.49% ALL and 69.75% PA
    test.plot_neurons('EM', output_filename='EM_predicted_with_jon_priors_optimize_all_predict.html')
    test.plot_neurons('clem', output_filename='CLEM_predicted_with_jon_priors_optimize_all_predict.html')

    test.predict_cells(use_jon_priors=False,suffix='_optimize_all_predict')
    test.plot_neurons('EM', output_filename='EM_predicted_optimize_all_predict.html')
    test.plot_neurons('clem', output_filename='CLEM_predicted_optimize_all_predict.html')

    test.calculate_verification_metrics(calculate_smat=True,with_kunst=False)

    #send_slack_message(MESSAGE='class_predictor.py finished!')