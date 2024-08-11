import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from cellpose import denoise
from cellpose import denoise
import cv2
import pandas as pd
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
from hindbrain_structure_function.visualization.FK_tools.load_pa_table import *
import re
from datetime import datetime
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.fragment_neurite import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.slack_bot import *
from matplotlib import colors
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette

import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm
from hindbrain_structure_function.functional_type_prediction.FK_tools.branching_angle_calculator import *

if __name__ == "__main__":
    def symmetric_log_transform(x, linthresh=1):
        return np.sign(x) * np.log1p(np.abs(x / linthresh))
    name_time = datetime.now()
    # set path
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')  # Ensure this path is set in path_configuration.txt
    #load em data
    all_cells_em = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["em"])
    all_cells_em = all_cells_em.sort_values('classifier')
    #load pa cells
    all_cells_pa = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"],use_smooth=False)
    all_cells_pa_smooth = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["pa"],use_smooth=True)
    all_cells_pa.loc[:,'swc_smooth'] = all_cells_pa_smooth['swc']
    all_cells_pa = all_cells_pa.dropna(subset='swc',axis=0)
    #load clem cells
    all_cells_clem = load_cells_predictor_pipeline(path_to_data=path_to_data, modalities=["clem"],load_repaired=True)
    all_cells_clem = all_cells_clem.dropna(subset='swc',axis=0)


    #prune
    all_cells_em.loc[:,'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_em['swc']]
    all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_pa['swc']]
    all_cells_pa.loc[:, 'swc_smooth'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_pa['swc_smooth']]
    all_cells_clem.loc[:, 'swc'] = [navis.prune_twigs(x, 5, recursive=True) for x in all_cells_clem['swc']]


    all_cells_em.loc[:,'swc'] = [navis.prune_twigs(x, 10, recursive=True) for x in all_cells_em['swc']]
    all_cells_pa.loc[:, 'swc'] = [navis.prune_twigs(x, 10, recursive=True) for x in all_cells_pa['swc']]
    all_cells_pa.loc[:, 'swc_smooth'] = [navis.prune_twigs(x, 10, recursive=True) for x in all_cells_pa['swc_smooth']]
    all_cells_clem.loc[:, 'swc'] = [navis.prune_twigs(x, 10, recursive=True) for x in all_cells_clem['swc']]

    #stack
    all_cells = pd.concat([all_cells_clem,all_cells_pa], axis=0)
    # all_cells = all_cells_clem
    all_cells = all_cells.reset_index(drop=True)

    #extract branching angle and coords of crossing for contralateral neurons
    for i,cell in tqdm(all_cells.iterrows(),total=all_cells.shape[0]):
        if cell.morphology == 'contralateral':
            #pass

            angle,crossing_coords,fragments_list = direct_angle_and_crossing_extraction(cell['swc'].nodes,projection="3d")
            angle2d, crossing_coords, fragments_list = direct_angle_and_crossing_extraction(cell['swc'].nodes, projection="2d")

            if np.isnan(angle):
                pass
            try:
                all_cells.loc[i,'angle'] = angle
                all_cells.loc[i, 'angle2d'] = angle2d
                all_cells.loc[i,'x_cross'] = crossing_coords[0]
                all_cells.loc[i,'y_cross'] = crossing_coords[1]
                all_cells.loc[i,'z_cross'] = crossing_coords[2]

            except:
                all_cells.loc[i, 'angle'] = np.nan
                all_cells.loc[i, 'angle2d'] = np.nan
                all_cells.loc[i, 'x_cross'] = np.nan
                all_cells.loc[i, 'y_cross'] = np.nan
                all_cells.loc[i, 'z_cross'] = np.nan

    send_slack_message(RECEIVER="Florian Kämpf", MESSAGE="decision_tree_minimal finished!")


    plt.figure(dpi=600)

    location_x_dict = {'nan': 0, "integrator": 1, 'dynamic threshold': 2, "motor command": 3, 'dynamic_threshold': 2, "motor_command": 3}
    location_x_mod = {'clem':-0.1,"photoactivation":0.1}
    color_dict_ec = {'clem':'black',"photoactivation":"orange"}
    color_dict = {
        'nan':"white",
        "integrator_ipsilateral": '#feb326b3',
        "integrator_contralateral": '#e84d8ab3',
        "dynamic_threshold": '#64c5ebb3',
        "dynamic threshold": '#64c5ebb3',
        "motor command": '#7f58afb3',
        "motor_command": '#7f58afb3'
    }
    from matplotlib.patches import Patch

    legend_elements = []
    for i, cell in all_cells.loc[~all_cells['angle'].isna(), :].iterrows():
        if cell['function'] == 'integrator':
            color_key =cell['function'] + "_" + cell['morphology']
        else:
            color_key = cell['function']
        plt.scatter(location_x_dict[cell['function']]+location_x_mod[cell['imaging_modality']], cell['angle'],c=color_dict[color_key],ec=color_dict_ec[cell['imaging_modality']],alpha=0.6)
        legend_name = color_key + " " + cell['imaging_modality']
        if not legend_name in [x.get_label() for  x in legend_elements]:
            legend_elements.append(Patch(facecolor=color_dict[color_key], edgecolor=color_dict_ec[cell['imaging_modality']],label=legend_name))

    plt.xticks([0, 1, 2, 3], ['na', 'ci', 'dt', 'mc'])


    plt.gca().legend(handles=legend_elements,frameon=False,fontsize = 'xx-small')

    plt.show()










    plt.figure(dpi=600)

    location_x_dict = {'nan': 0, "integrator": 1, 'dynamic threshold': 2, "motor command": 3, 'dynamic_threshold': 2, "motor_command": 3}
    location_x_mod = {'clem':-0.1,"photoactivation":0.1}
    color_dict_ec = {'clem':'black',"photoactivation":"orange"}
    color_dict = {
        'nan':"white",
        "integrator_ipsilateral": '#feb326b3',
        "integrator_contralateral": '#e84d8ab3',
        "dynamic_threshold": '#64c5ebb3',
        "dynamic threshold": '#64c5ebb3',
        "motor command": '#7f58afb3',
        "motor_command": '#7f58afb3'
    }
    from matplotlib.patches import Patch

    legend_elements = []
    for i, cell in all_cells.loc[~all_cells['angle'].isna(), :].iterrows():
        if cell['function'] == 'integrator':
            color_key =cell['function'] + "_" + cell['morphology']
        else:
            color_key = cell['function']
        plt.scatter(location_x_dict[cell['function']]+location_x_mod[cell['imaging_modality']], cell['angle2d'],c=color_dict[color_key],ec=color_dict_ec[cell['imaging_modality']],alpha=0.6)
        legend_name = color_key + " " + cell['imaging_modality']
        if not legend_name in [x.get_label() for  x in legend_elements]:
            legend_elements.append(Patch(facecolor=color_dict[color_key], edgecolor=color_dict_ec[cell['imaging_modality']],label=legend_name))

    plt.xticks([0, 1, 2, 3], ['na', 'ci', 'dt', 'mc'])
    plt.title('2d')

    plt.gca().legend(handles=legend_elements,frameon=False,fontsize = 'xx-small')

    plt.show()













    #location
    legend_elements = []
    for i, cell in all_cells.loc[~all_cells['angle'].isna(), :].iterrows():
        if cell['function'] == 'integrator':
            color_key =cell['function'] + "_" + cell['morphology']
        else:
            color_key = cell['function']
        if color_key != "nan":
            plt.scatter(cell['y_cross'],cell['z_cross'],c=color_dict[color_key],ec=color_dict_ec[cell['imaging_modality']],alpha=0.9)
            legend_name = color_key + " " + cell['imaging_modality']
            if not legend_name in [x.get_label() for  x in legend_elements]:
                legend_elements.append(Patch(facecolor=color_dict[color_key], edgecolor=color_dict_ec[cell['imaging_modality']],label=legend_name))




    plt.gca().legend(handles=legend_elements,frameon=False,fontsize = 'xx-small')
    plt.xlabel('Y-AXIS')
    plt.ylabel('Z-AXIS')

    plt.xticks([plt.xlim()[0], plt.xlim()[1]], ['Rostral','Caudal'])
    plt.yticks([plt.ylim()[0], plt.ylim()[1]], ['Ventral', 'Dorsal'])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.title('location crossing')
    plt.show()





    #prediction

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from metric_learn import LMNN
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    # all the values
    #complete_dataset = all_cells.loc[(all_cells['function'] != 'nan')&(all_cells['imaging_modality'] == 'photoactivation'), :].dropna(subset = ['angle', 'angle2d','x_cross', 'y_cross', 'z_cross'])
    complete_dataset = all_cells.loc[(all_cells['function'] != 'nan'), :].dropna(subset=['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross','function','morphology'])




    complete_dataset.loc[:, 'function'] = complete_dataset.loc[:, ['morphology', 'function']].apply(lambda x: x['function'].replace('_', " "), axis=1)

    features_df = complete_dataset.loc[:,['angle', 'angle2d','x_cross', 'y_cross', 'z_cross']]
    features = np.array(features_df)
    labels = np.array(complete_dataset.loc[:, ['function']])




    # split

    X_train, X_test, y_train, y_test = train_test_split(features, labels,stratify=labels, test_size=0.7, random_state=42)



    nca = NeighborhoodComponentsAnalysis(random_state=42,n_components=2,init='lda')
    nca.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=5,weights='distance')
    knn.fit(X_train, y_train)


    print(knn.score(X_test, y_test))

    knn.fit(nca.transform(X_train), y_train)

    print(knn.score(nca.transform(X_test), y_test))



    #lmnn
    lmnn = LMNN(n_neighbors=3, learn_rate=1e-8,min_iter=1000,regularization=0.2,n_components=3)
    lmnn.fit(X_train, y_train)


    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_train, y_train)

    print(knn.score(X_test, y_test))
    knn.fit(lmnn.transform(X_train), y_train)
    print(knn.score(lmnn.transform(X_test), y_test))

    #LDS

    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)
    correct = prediction == y_test.flatten()
    percent_correct = correct.sum() / len(correct)
    print(percent_correct)


    #pca

    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    color_dict_ec = {'clem':'black',"photoactivation":"orange"}
    color_dict = {
        'nan':"white",
        "integrator": '#feb326b3',
        "integrator_ipsilateral": '#feb326b3',
        "integrator_contralateral": '#e84d8ab3',
        "dynamic_threshold": '#64c5ebb3',
        "dynamic threshold": '#64c5ebb3',
        "motor command": '#7f58afb3',
        "motor_command": '#7f58afb3'
    }



    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    pca = PCA(n_components=2)

    principal_components = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df.loc[:,'function'] = list(complete_dataset.loc[:,'function'])
    pca_df.loc[:,'morphology'] = list(complete_dataset['morphology'])
    pca_df.loc[:,'imaging_modality'] = list(complete_dataset['imaging_modality'])

    print(pca_df)
    legend_elements = []
    plt.figure(figsize=(10, 10))
    for i,cell in pca_df.iterrows():
        plt.scatter(cell['PC1'], cell['PC2'], c=color_dict[cell['function']],ec=color_dict_ec[cell['imaging_modality']],alpha=0.6)

        legend_name = cell['function'] + " " + cell['imaging_modality']
        if not legend_name in [x.get_label() for x in legend_elements]:
            legend_elements.append(Patch(facecolor=color_dict[cell['function']], edgecolor=color_dict_ec[cell['imaging_modality']], label=legend_name))

    plt.gca().legend(handles=legend_elements, frameon=False, fontsize='xx-small')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Sample Data')
    plt.grid()
    plt.show()
    temp = all_cells.loc[:, ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross']]
    temp.to_hdf(path_to_data / 'make_figures_FK_output' / 'CLEM_and_PA_features.hdf5', 'angle_cross')


