import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
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

repeats = 10000
path_to_data = get_base_path()
# LDS
fmn = pd.read_hdf(path_to_data / 'make_figures_FK_output' / 'CLEM_and_PA_features.hdf5','function_morphology_neurotransmitter')
pp = pd.read_hdf(path_to_data / 'make_figures_FK_output' / 'CLEM_and_PA_features.hdf5','predictor_pipeline_features')
ac = pd.read_hdf(path_to_data / 'make_figures_FK_output' / 'CLEM_and_PA_features.hdf5','angle_cross')

all_cells = pd.concat([fmn,pp,ac],axis=1)

# data
without_nan_function = all_cells.loc[(all_cells['function'] != 'nan'), :]

#impute nans
columns_possible_nans = ['angle','angle2d','x_cross','y_cross','z_cross']
for work_column in columns_possible_nans:

        without_nan_function.loc[without_nan_function[work_column].isna(), work_column] = 0


#replace strings
columns_replace_string = ['neurotransmitter']
for work_column in columns_replace_string:
    for i,unique_feature in enumerate(without_nan_function[work_column].unique()):
        without_nan_function.loc[without_nan_function[work_column] == unique_feature, work_column] = i





features = np.array(without_nan_function.loc[:, without_nan_function.columns[4:]])
features_pa = np.array(without_nan_function.loc[without_nan_function['imaging_modality'] == 'photoactivation', without_nan_function.columns[4:]])
features_clem = np.array(without_nan_function.loc[without_nan_function['imaging_modality'] == 'clem', without_nan_function.columns[4:]])





scaler = StandardScaler()
features = scaler.fit_transform(features)
features_pa = scaler.fit_transform(features_pa)
features_clem = scaler.fit_transform(features_clem)

without_nan_function.loc[:, 'function'] = without_nan_function.loc[:, ['morphology', 'function']].apply(lambda x: x['function'].replace('_', " "), axis=1)
without_nan_function_pa = copy.deepcopy(without_nan_function.loc[without_nan_function['imaging_modality'] == 'photoactivation', :])
without_nan_function_pa.loc[:, 'function'] = without_nan_function_pa.loc[:, ['morphology', 'function']].apply(lambda x: x['function'].replace('_', " "), axis=1)
without_nan_function_clem = copy.deepcopy(without_nan_function.loc[without_nan_function['imaging_modality'] == 'clem', :])
without_nan_function_clem.loc[:, 'function'] = without_nan_function_clem.loc[without_nan_function_clem['imaging_modality'] == 'clem', ['morphology', 'function']].apply(lambda x: x['function'].replace('_', " "), axis=1)

for i, cell in without_nan_function.iterrows():
    if cell['function'] == 'integrator':
        without_nan_function.loc[i, 'function'] = cell['function'] + " " + cell['morphology']
for i, cell in without_nan_function_clem.iterrows():
            if cell['function'] == 'integrator':
                without_nan_function_clem.loc[i, 'function'] = cell['function'] + " " + cell['morphology']

for i, cell in without_nan_function_pa.iterrows():
    if cell['function'] == 'integrator':
        without_nan_function_pa.loc[i, 'function'] = cell['function'] + " " + cell['morphology']



labels = np.array(without_nan_function.loc[:, ['function']])
labels_pa = np.array(without_nan_function_pa.loc[:, ['function']])
labels_clem = np.array(without_nan_function_clem.loc[:, ['function']])

# LDA
rs = 42
collection_coef_matrix = None
shuffled_labels =  False
collection_prediction_correct = []
solver = 'lsqr'
shrinkage = 'auto'

X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=0.3, random_state=rs)

# Create the LDA model
clf = SVC(probability=True)
clf2 = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

# Fit the model
clf.fit(X_train, y_train.flatten())
clf2.fit(X_train, y_train.flatten())
if shuffled_labels:
    clf.fit(X_train,np.random.permutation(y_train.flatten())) #random labels

# Make predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

y_pred2 = clf2.predict(X_test)
y_prob2 = clf2.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy2 = accuracy_score(y_test, y_pred2)



collection_prediction_correct.append(accuracy)


print('Classification Report:')
print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred2))
print(f'Accuracy: {round(accuracy * 100, 4)}%')




for i in tqdm(range(repeats), total=repeats):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels, test_size=0.3, random_state=rs)

    # Create the LDA model
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    # Fit the model
    clf.fit(X_train, y_train.flatten())
    if shuffled_labels:
        clf.fit(X_train, np.random.permutation(y_train.flatten()))  # random labels

    # Make predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate the percentage of correct predictions
    collection_prediction_correct.append(accuracy)

    # print(f'Accuracy: {accuracy}')
    # print('Classification Report:')
    # print(classification_report(y_test, y_pred))
    # print(f'Percentage Correct: {round(percent_correct,4)*100}%\n')

    coef_matrix = clf.coef_
    if collection_coef_matrix is None:
        collection_coef_matrix = coef_matrix[np.newaxis, :, :]
    else:
        collection_coef_matrix = np.vstack([collection_coef_matrix, coef_matrix[np.newaxis, :, :]])
    rs += 1

print(f"All Features Mean accuracy over {repeats} repeats: {round(np.mean(collection_prediction_correct)*100,2)}%")
print(f"All Features Max accuracy  over {repeats} repeats: {round(np.max(collection_prediction_correct)*100,2)}%")
print(f"All Features Min accuracy over {repeats} repeats: {round(np.min(collection_prediction_correct)*100,2)}%")
print(f"All Features Std of accuracy over {repeats} repeats: {round(np.std(collection_prediction_correct)*100,2)}%")

# rerun model with only the features that have a 2weight above 0.5
coef_matrix_avg = np.mean(collection_coef_matrix, axis=0)

plt.figure(figsize=(15, 15))
column_labels = list(without_nan_function.columns[4:])
aaa = plt.pcolormesh(coef_matrix_avg)
plt.xticks(np.arange(0.5, coef_matrix.shape[1]), column_labels, rotation=90, fontsize='x-small')
plt.subplots_adjust(left=0.3, right=0.8, top=0.5, bottom=0.3)
plt.title(f'average weights {repeats} repeats')
plt.colorbar(aaa)
plt.show()

features_with_high_weights_bool = np.sum((abs(coef_matrix) > 0.5), axis=0).astype(bool)
print(f"{len(list(np.array(column_labels)[features_with_high_weights_bool]))} Features used\n", list(np.array(column_labels)[features_with_high_weights_bool]), '\n')

# LDA
rs = 42


rs = 42
collection_coef_matrix = None
collection_prediction_correct = []

X_train, X_test, y_train, y_test = train_test_split(features[:, features_with_high_weights_bool], labels, stratify=labels, test_size=0.3, random_state=rs)

# Create the LDA model
clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

# Fit the model
clf.fit(X_train, y_train.flatten())

# Make predictions
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)





print('Classification Report:')
print(classification_report(y_test, y_pred))
print(f'Accuracy: {round(accuracy * 100, 4)}%')




collection_prediction_correct_small = []

for i in tqdm(range(repeats), total=repeats):
    X_train, X_test, y_train, y_test = train_test_split(features[:, features_with_high_weights_bool], labels, stratify=labels, test_size=0.3, random_state=rs)
    # Create the LDA model
    clf = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    # Fit the model
    clf.fit(X_train, y_train.flatten())

    # Make predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate the percentage of correct predictions
    prediction = clf.predict(X_test)
    correct = prediction == y_test.flatten()
    percent_correct = correct.sum() / len(correct)
    # print(f'Percentage Correct: {round(percent_correct,4)*100}%\n')
    # print(f'Accuracy: {accuracy}')
    # print('Classification Report:')
    # print(classification_report(y_test, y_pred))
    collection_prediction_correct_small.append(percent_correct)
    rs += 1


print(f"Reduced Features Mean accuracy over {repeats} repeats: {round(np.mean(collection_prediction_correct_small)*100,2)}%")
print(f"Reduced Features Max accuracy  over {repeats} repeats: {round(np.max(collection_prediction_correct_small)*100,2)}%")
print(f"Reduced Features Min accuracy over {repeats} repeats: {round(np.min(collection_prediction_correct_small)*100,2)}%")
print(f"Reduced Features Std of accuracy over {repeats} repeats: {round(np.std(collection_prediction_correct_small)*100,2)}%")



for i,cell in without_nan_function.iterrows():
    if cell['function'] == 'integrator':
        without_nan_function.loc[i, 'label'] = cell['function'].replace('_'," ") + " " + cell['morphology']
    else:
        without_nan_function.loc[i, 'label'] = cell['function'].replace('_'," ")

no_plots = int(np.ceil(np.sqrt(len(np.array(column_labels)[features_with_high_weights_bool]))))

fig,ax = plt.subplots(nrows=no_plots, ncols=no_plots,figsize=(15,15))
x=0
y=0
legend_elements = []
color_dict = {
    'nan': "white",
    "integrator ipsilateral": '#feb326b3',
    "integrator contralateral": '#e84d8ab3',
    "dynamic_threshold": '#64c5ebb3',
    "dynamic threshold": '#64c5ebb3',
    "motor command": '#7f58afb3',
    "motor_command": '#7f58afb3'
}
color_dict_ec = {'clem':'black',"photoactivation":"orange"}

for feature in list(np.array(column_labels)[features_with_high_weights_bool]):
    for label in np.unique(labels):
        temp_label_df = without_nan_function.loc[(without_nan_function['label'] == label)&(without_nan_function['imaging_modality'] == 'clem'),:]
        kkk = sns.kdeplot(list(temp_label_df[feature]),label = label,fill=False,ax=ax[x,y],color=color_dict[label])
        if not label in [x.get_label() for  x in legend_elements]:
            legend_elements.append(Patch(facecolor=color_dict[label],label=label))
    #plt.legend(frameon=False,fontsize='x-small')
    ax[x, y].set_title(feature, fontsize='x-small')
    x+=1
    if x == no_plots:
        x=0
        y+=1
    ax[-1, -1].legend(handles=legend_elements,frameon=False,fontsize = 'small')

for x in range(no_plots):
    for y in range(no_plots):

        ax[x, y].set_xlabel('')
        ax[x, y].set_ylabel('')
        ax[x, y].set_yticklabels('')
        ax[x, y].set_yticks([])
        ax[x, y].spines[['right', 'top', "left"]].set_visible(False)
plt.show()








feature1 = features[:, features_with_high_weights_bool]
feature2 = features_clem[:, features_with_high_weights_bool]
feature3 = features_pa[:, features_with_high_weights_bool]
fig,ax = plt.subplots(nrows=3,figsize=(5,15))
for data,mode,l,i in zip([feature1, feature2, feature3],['both','clem','pa'],[labels,labels_clem,labels_pa],range(3)):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)


    pca = PCA(n_components=2)

    principal_components = pca.fit_transform(scaled_data)


    sns.kdeplot(x=principal_components[:,0],y=principal_components[:,1],hue=l.flatten(),ax=ax[i])
    ax[i].set_title(f'PCA {mode}')
    ax[i].set_aspect('equal', 'box')

plt.show()


#3d pca
fig = None
for data,mode,l,i in zip([feature1, feature2, feature3],['both','clem','pa'],[labels,labels_clem,labels_pa],range(3)):
    temp_label = l
    temp_features = data

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(temp_features)

    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(scaled_data)

    import plotly
    import navis



    for it_label in np.unique(temp_label):
        temp_points = principal_components[temp_label.flatten()==it_label,:]
        temp_point_cloud = tm.points.PointCloud(temp_points)
        temp_convex_hull = temp_point_cloud.convex_hull

        temp_volume = navis.MeshNeuron(temp_convex_hull)
        temp_volume.name = f'{mode} {it_label}'


        if type(fig) == type(None):
            fig = navis.plot3d(temp_volume, backend='plotly', color=color_dict[it_label],
                               width=1920, height=1080, hover_name=True,alpha=0.3)
        else:
            fig = navis.plot3d(temp_volume, backend='plotly', color=color_dict[it_label],fig=fig,
                               width=1920, height=1080, hover_name=True,alpha=0.3)


fig.update_layout(
    scene={
        'xaxis': {'autorange': 'reversed'},  # reverse !!!
        'yaxis': {'autorange': True},

        'zaxis': {'autorange': True},
        'aspectmode': "data",
        'aspectratio': {"x": 1, "y": 1, "z": 1}
    }
)

plotly.offline.plot(fig, filename="test.html", auto_open=True, auto_play=False)