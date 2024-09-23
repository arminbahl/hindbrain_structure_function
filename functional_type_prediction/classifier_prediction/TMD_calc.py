import tmd
from pathlib import Path
import os
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
import tmd
import navis
import pandas as pd
from tmd.utils import SOMA_TYPE
from tmd.Soma import Soma
from tmd.utils import TREE_TYPE_DICT
import os
from tmd.io.h5 import read_h5
from tmd.utils import TmdError
import warnings
import numpy as np
from tmd.io.swc import SWC_DCT
from scipy import sparse as sp
from scipy.sparse import csgraph as cs
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.calculate_metric2df_semiold import *

np.set_printoptions(suppress=True)
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

if __name__ == "__main__":
    # New segment: set constants
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')
    path_to_save = path_to_data / 'make_figures_FK_output' / 'TMD'
    os.makedirs(path_to_save, exist_ok=True)
    n_estimators_rf = 100
    use_new_neurotransmitter = True
    use_k_means_classes = True

    # New segment: load FK_features
    all, column_labels, all_cells = load_metrics_train('FINAL', path_to_data=path_to_data)  # clem_clem_predict_pa_prediction_project_neg_controls

    # unpack metrics
    features_fk, labels_fk, labels_imaging_modality = all

    # New segment: load all cells and calculate persistence_vectors, persistence_samples, form_factor
    cells = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['pa', 'clem'], load_repaired=True)
    cells = prepare_data_4_metric_calc(cells, use_new_neurotransmitter, use_k_means_classes, path_to_data=path_to_data)
    cells = cells.set_index('cell_name').loc[all_cells['cell_name']].reset_index()
    cells['swc'] = cells['swc'].apply(lambda x: x.resample("1 micron"))



    cells['class'] = cells.loc[:, ['function', 'morphology']].apply(lambda x: x['function'].replace(" ", "_") + "_" + x['morphology'] if x['function'] == 'integrator' else x['function'].replace(" ", "_"), axis=1)
    labels = cells['class'].to_numpy()

    clem_idx = (cells['imaging_modality'] == 'clem').to_numpy()
    pa_idx = (cells['imaging_modality'] == 'photoactivation').to_numpy()


    # New segment: TMD aka Armins paper https://link.springer.com/article/10.1007/s12021-017-9341-1
    path = r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\paGFP\all_cells_repaired\20230413.2_repaired.swc"
    path = r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\clem_zfish1\all_cells_repaired\clem_zfish1_cell_576460752488813678_repaired.swc"
    # path = r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\make_figures_FK_output\C010398B-P2.CNG.swc"


    neu = tmd.io.load_neuron(path)
    neu = neu.simplify()
    # pd = tmd.methods.get_persistence_diagram(neu.neurites[0])



    #New segment: example code from https://github.com/BlueBrain/TMD/blob/master/examples/extract_ph.py

    # Step 1: Import the tmd module
    import tmd
    from tmd.view import plot
    from tmd.view import view


    # Step 2: Load your morphology
    filename = r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\make_figures_FK_output\C010398B-P2.CNG.swc"
    filename = r"C:\Users\ag-bahl\Downloads\20230413.2_repaired_modified.soma.swc"
    neu = tmd.io.load_neuron(filename)

    # Step 3: Extract the ph diagram of a tree
    tree = neu.neurites[0]
    ph = tmd.methods.get_persistence_diagram(tree)

    # Step 4: Extract the ph diagram of a neuron's trees
    ph_neu = tmd.methods.get_ph_neuron(neu)

    # Step 5: Extract the ph diagram of a neuron's trees,
    # depending on the neurite_type
    ph_apical = tmd.methods.get_ph_neuron(neu, neurite_type="apical_dendrite")
    ph_axon = tmd.methods.get_ph_neuron(neu, neurite_type="axon")
    ph_basal = tmd.methods.get_ph_neuron(neu, neurite_type="basal_dendrite")

    # Step 6: Plot the extracted topological data with three different ways

    # Visualize the neuron
    view.neuron(neu)
    plt.show()
    # Visualize a selected neurite type or multiple of them
    view.neuron(neu, neurite_type=["apical_dendrite"])
    plt.show()
    # Visualize the persistence diagram
    plot.diagram(ph_apical)
    plt.show()
    # Visualize the persistence barcode
    plot.barcode(ph_apical)
    plt.show()
    # Visualize the persistence image
    plot.persistence_image(ph_apical)
    plt.show()


