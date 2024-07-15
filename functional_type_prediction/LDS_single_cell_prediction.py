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
import seaborn as sns
import trimesh as tm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy

def generate_matching_plot(features,labels,labels_imaging_modality,path,solver='lsqr',shrinkage='auto',title='prediction_plot'):
    #init variables
    prob_matrix = np.empty(shape=(features.shape[0],len(np.unique(labels))))
    pred_matrix = np.empty(shape=(features.shape[0],1),dtype='<U24')
    prediction_correct = []
    legend_elements = []

    #loop over cells
    for i in range(features.shape[0]):
        X_train = features[[x for x in range(features.shape[0]) if x != i]]
        X_test = features[i,:]
        y_train = labels[[x for x in range(features.shape[0]) if x != i]]
        y_test = labels[i]

        #create and fit lda
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        clf.fit(X_train, y_train.flatten())

        # predict
        y_pred = clf.predict(X_test[np.newaxis,:])
        y_prob = clf.predict_proba(X_test[np.newaxis,:])
        prediction_correct.append(y_pred==y_test)

        prob_matrix[i,:] = y_prob
        pred_matrix[i,:] = y_pred


    print(f'Predictions correct: {np.round((np.sum(prediction_correct)/len(prediction_correct)*100),2)}%')
    #figure part

    color_dict_type = {
        "integrator ipsilateral": '#feb326b3',
        "integrator contralateral": '#e84d8ab3',
        "dynamic threshold": '#64c5ebb3',
        "motor command": '#7f58afb3',
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
        y = location_dict[item[0]]
        plt.plot([x,x],[y,y+1],lw=2,color='red')
        plt.plot([x, x+1], [y+1, y + 1], lw=2, color='red')
        plt.plot([x, x + 1], [y, y ], lw=2, color='red')
        plt.plot([x+1 , x+1], [y, y + 1], lw=2, color='red')

    ax.set_yticks(np.arange(len(labels_sort)) + 0.5, labels_sort)
    ax.set_xlim(-2, len(features))
    ax.set_ylim(-2, len(labels_sort))

    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    fig.colorbar(im, orientation='vertical')
    ax.set_xticks([])

    savepath = path_to_data / 'make_figures_FK_output' / 'LDA_cell_type_prediction'
    os.makedirs(savepath,exist_ok=True)
    fig.set_dpi(450)
    ax.legend(handles=legend_elements, frameon=False)
    plt.title(title + f'\nPredictions correct: {np.round((np.sum(prediction_correct) / len(prediction_correct) * 100), 2)}%')
    plt.savefig(savepath / (title.replace('\n'," ") + ".pdf"))
    plt.savefig(savepath / (title.replace('\n'," ")+ ".png"))




    plt.show()
def generate_matching_plot_test_and_train_not_the_same(features_train,labels_train,features_test,labels_test,labels_imaging_modality,path,solver='lsqr',shrinkage='auto',title='prediction_plot'):
    #init variables
    prob_matrix = np.empty(shape=(features_test.shape[0],len(np.unique(labels))))
    pred_matrix = np.empty(shape=(features_test.shape[0],1),dtype='<U24')
    prediction_correct = []
    legend_elements = []

    #TRAIN
    X_train = features_train
    y_train = labels_train

    # create and fit lda
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    clf.fit(X_train, y_train.flatten())


    #loop over cells
    for i in range(features_test.shape[0]):

        X_test = features_test[i,:]

        y_test = labels_test[i]

        # predict
        y_pred = clf.predict(X_test[np.newaxis,:])
        y_prob = clf.predict_proba(X_test[np.newaxis,:])
        prediction_correct.append(y_pred==y_test)

        prob_matrix[i,:] = y_prob
        pred_matrix[i,:] = y_pred


    print(f'Predictions correct: {np.round((np.sum(prediction_correct)/len(prediction_correct)*100),2)}%')
    #figure part

    color_dict_type = {
        "integrator ipsilateral": '#feb326b3',
        "integrator contralateral": '#e84d8ab3',
        "dynamic threshold": '#64c5ebb3',
        "motor command": '#7f58afb3',
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
        y = location_dict[item[0]]
        plt.plot([x,x],[y,y+1],lw=2,color='red')
        plt.plot([x, x+1], [y+1, y + 1], lw=2, color='red')
        plt.plot([x, x + 1], [y, y ], lw=2, color='red')
        plt.plot([x+1 , x+1], [y, y + 1], lw=2, color='red')

    ax.set_yticks(np.arange(len(labels_sort)) + 0.5, labels_sort)
    ax.set_xlim(-2, len(features_test))
    ax.set_ylim(-2, len(labels_sort))

    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    fig.colorbar(im, orientation='vertical')
    ax.set_xticks([])

    savepath = path_to_data / 'make_figures_FK_output' / 'LDA_cell_type_prediction'
    os.makedirs(savepath,exist_ok=True)
    fig.set_dpi(450)
    ax.legend(handles=legend_elements, frameon=False)
    plt.title(title + f'\nPredictions correct: {np.round((np.sum(prediction_correct) / len(prediction_correct) * 100), 2)}%')
    plt.savefig(savepath / (title.replace('\n'," ") + ".pdf"))
    plt.savefig(savepath / (title.replace('\n'," ")+ ".png"))




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
def determine_important_features(features,labels,feature_labels, repeats=10000,random_seed=42,solver='lsqr',shrinkage='auto',test_size=0.3,stratify=True):
    #init variables
    collection_coef_matrix = None
    collection_prediction_correct = []
    #sort features and labels in a predetermined way
    temp_array = np.hstack([features,labels[:,np.newaxis]])
    rng = np.random.default_rng(seed=random_seed)
    temp_array = rng.permutation(temp_array)
    features = temp_array[:,:-1]
    labels = temp_array[:,-1]


    for i in tqdm(range(repeats)):
        #split_data
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=test_size, random_state=random_seed)
        else:
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_seed)

        # Create the LDA model
        clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
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



    return reduced_features,reduced_features_bool




# Constants
repeats = 10000
path_to_data = get_base_path()

# Data Loading
file_path = path_to_data / 'make_figures_FK_output' / 'CLEM_and_PA_features.hdf5'
fmn = pd.read_hdf(file_path, 'function_morphology_neurotransmitter')
pp = pd.read_hdf(file_path, 'predictor_pipeline_features')
ac = pd.read_hdf(file_path, 'angle_cross')

all_cells = pd.concat([fmn, pp, ac], axis=1)

# Data Preprocessing
without_nan_function = all_cells[all_cells['function'] != 'nan']

# Impute NaNs
columns_possible_nans = ['angle', 'angle2d', 'x_cross', 'y_cross', 'z_cross']
without_nan_function.loc[:,columns_possible_nans] = without_nan_function[columns_possible_nans].fillna(0)

# Replace strings with indices
columns_replace_string = ['neurotransmitter']
for work_column in columns_replace_string:
    for i,unique_feature in enumerate(without_nan_function[work_column].unique()):
        without_nan_function.loc[without_nan_function[work_column] == unique_feature, work_column] = i

# Function string replacement
without_nan_function.loc[:,'function'] = without_nan_function['function'].str.replace('_', ' ')
without_nan_function_pa = without_nan_function[without_nan_function['imaging_modality'] == 'photoactivation'].copy()
without_nan_function_pa.loc[:,'function'] = without_nan_function_pa['function'].str.replace('_', ' ')
without_nan_function_clem = without_nan_function[without_nan_function['imaging_modality'] == 'clem'].copy()
without_nan_function_clem.loc[:,'function'] = without_nan_function_clem['function'].str.replace('_', ' ')

# Update 'integrator' function
def update_integrator(df):
    integrator_mask = df['function'] == 'integrator'
    df.loc[integrator_mask, 'function'] += " " + df.loc[integrator_mask, 'morphology']

update_integrator(without_nan_function)
update_integrator(without_nan_function_pa)
update_integrator(without_nan_function_clem)

#sort by function an imaging modality
without_nan_function = without_nan_function.sort_values(by=['function','morphology','imaging_modality','neurotransmitter'])
without_nan_function_pa = without_nan_function_pa.sort_values(by=['function','imaging_modality','morphology','neurotransmitter'])
without_nan_function_clem = without_nan_function_clem.sort_values(by=['function','imaging_modality','morphology','neurotransmitter'])

# Extract labels
labels = without_nan_function['function'].to_numpy()
labels_imaging_modality = without_nan_function['imaging_modality'].to_numpy()
labels_imaging_modality_pa = without_nan_function_pa['imaging_modality'].to_numpy()
labels_imaging_modality_clem = without_nan_function_clem['imaging_modality'].to_numpy()
labels_pa = without_nan_function_pa['function'].to_numpy()
labels_clem = without_nan_function_clem['function'].to_numpy()
column_labels = list(without_nan_function.columns[3:])

# Extract features
features = without_nan_function.iloc[:, 3:].to_numpy()
features_pa = without_nan_function[without_nan_function['imaging_modality'] == 'photoactivation'].iloc[:, 3:].to_numpy()
features_clem = without_nan_function[without_nan_function['imaging_modality'] == 'clem'].iloc[:, 3:].to_numpy()

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)
features_pa = scaler.transform(features_pa)
features_clem = scaler.transform(features_clem)


# LDA

reduced_features,reduced_features_index = determine_important_features(features,labels,column_labels)

#BOTH TRAINING AND TESTING
generate_matching_plot(features,labels,labels_imaging_modality,path=path_to_data,title='All features\nTrained on CLEM & PA\nTested on CLEM & PA')
generate_matching_plot(features[:,reduced_features_index],labels,labels_imaging_modality,path=path_to_data,title='Reduced features\nTrained on CLEM & PA\nTested on CLEM & PA')

#CLEM TRAINING AND PA TESTING
generate_matching_plot_test_and_train_not_the_same(features_clem,labels_clem,features_pa,labels_pa,labels_imaging_modality_pa,path=path_to_data,
                                                   title='All features\nTrained on CLEM\nTested on PA')
generate_matching_plot_test_and_train_not_the_same(features_clem[:,reduced_features_index],labels_clem,features_pa[:,reduced_features_index],labels_pa,labels_imaging_modality_pa,path=path_to_data,
                       title='Reduced features\nTrained on CLEM\nTested on PA')

#CLEM TRAINING AND CLEM TESTING
generate_matching_plot(features_clem,labels_clem,labels_imaging_modality_clem,path=path_to_data,title='All features\nTrained on CLEM\nTested on CLEM')
generate_matching_plot(features_clem[:,reduced_features_index],labels_clem,labels_imaging_modality_clem,path=path_to_data,title='Reduced features\nTrained on CLEM\nTested on CLEM')

#PA TRAINING AND CLEM TESTING
generate_matching_plot_test_and_train_not_the_same(features_pa,labels_pa,features_clem,labels_clem,labels_imaging_modality_clem,path=path_to_data,
                                                   title='All features\nTrained on PA\nTested on CLEM')
generate_matching_plot_test_and_train_not_the_same(features_pa[:,reduced_features_index],labels_pa,features_clem[:,reduced_features_index],labels_clem,labels_imaging_modality_clem,path=path_to_data,
                       title='Reduced features\nTrained on PA\nTested on CLEM')

#PA TRAINING AND PA TESTING
generate_matching_plot(features_pa,labels_pa,labels_imaging_modality_pa,path=path_to_data,title='All features\nTrained on PA\nTested on PA')
generate_matching_plot(features_pa[:,reduced_features_index],labels_pa,labels_imaging_modality_pa,path=path_to_data,title='Reduced features\nTrained on PA\nTested on PA')





