import matplotlib.pyplot as plot
import seaborn as sns
import pandas as pd
from hindbrain_structure_function.functional_type_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.functional_type_prediction.LDS_predict_jon_cells import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
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
from hindbrain_structure_function.functional_type_prediction.FK_tools.slack_bot import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm
from hindbrain_structure_function.functional_type_prediction.FK_tools.branching_angle_calculator import *
import copy
import os
from hindbrain_structure_function.functional_type_prediction.nblast_matrix_navis import *
import matplotlib.pyplot as plt
import scipy
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.functional_type_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.make_dendrogramms import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from hindbrain_structure_function.visualization.make_figures_FK import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.find_branches import *
import winsound
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import plotly
import matplotlib
# matplotlib.use('TkAgg')
from tqdm import tqdm
from sklearn.svm import OneClassSVM
import winsound

if __name__ == '__main__':
    # set variables
    np.set_printoptions(suppress=True)
    path_to_data = get_base_path()
    brain_meshes = load_brs(path_to_data, 'raphe')
    width_brain = 495.56
    path_to_figure_dir = path_to_data / 'make_figures_FK_output' / 'matched_cells_all_modalities'
    os.makedirs(path_to_figure_dir,exist_ok=True)


    master_df = pd.read_excel(path_to_data / 'make_figures_FK_output' / 'find_match_cell_EM_CLEM_paGFP' / 'matching_cells.xlsx',dtype=str)
    dt_df = master_df.iloc[:,[1,2,3]]
    mc_df = master_df.iloc[:, [6, 7, 8]]
    ii_df = master_df.iloc[:, [11, 12, 13]]
    ci_df = master_df.iloc[:, [16, 17, 18]]

    for df in [dt_df, mc_df, ii_df, ci_df]:
        for i,row in df.iterrows():
            if type(row.iloc[0]) == float:
                df.iloc[i,0] = df.iloc[i-1,0]
            if type(row.iloc[1]) == float:
                df.iloc[i,1] = df.iloc[i-1,1]

    dict2abrv = {'integrator ipsilateral':'II','integrator contralateral':'CI','motor command':'MC',"dynamic threshold":'DT'}

    #load cells

    cells_em = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['em'], load_repaired=False,load_both=True,mirror=True)
    cells_em = cells_em.reset_index(drop=True)
    cells_em = cells_em.sort_values('cell_name')
    cells_em['soma_mesh'] = cells_em['soma_mesh'].apply(lambda x: navis.simplify_mesh(x, 100))

    cells_clem = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem'], load_repaired=False, load_both=True, mirror=True)
    cells_clem = cells_clem.loc[(cells_clem['function'] != 'nan') | (cells_clem['function'].isna()), :]
    cells_clem = cells_clem.reset_index(drop=True)
    cells_clem = cells_clem.sort_values('cell_name')

    cells_clem['soma_mesh'] = cells_clem['soma_mesh'].apply(lambda x: navis.simplify_mesh(x,100))

    cells_clem['class'] = cells_clem.apply(lambda x: dict2abrv[x.function.replace('_', ' ')] if x.function != 'integrator' else dict2abrv[x.function + " " + x.morphology], axis=1)

    cells_pa = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['pa'], load_repaired=False, load_both=True,
                                             mirror=True)
    cells_pa = cells_pa.reset_index(drop=True)

    cells_pa['class'] = cells_pa.apply(lambda x: dict2abrv[x.function.replace('_',' ')] if x.function != 'integrator' else dict2abrv[x.function +" "+ x.morphology],axis=1 )
    cells_pa = cells_pa.sort_values('cell_name')

    #cehck if there are wrong entries
    all_real_cells = np.concatenate([cells_clem.cell_name.to_numpy(), cells_em.cell_name.to_numpy(), cells_pa.cell_name.to_numpy()]).flatten()
    all_real_cells = [x[5:] if 'cell' in x else x for x in all_real_cells]
    all_excel_values = master_df.to_numpy().flatten()
    all_excel_values = np.array([x for x in all_excel_values if x is not np.nan])
    indexer = [not x for x in  np.isin(all_excel_values, all_real_cells)]
    print('All values not real names: ', list(np.unique(all_excel_values[indexer])))


    for df,cell_class in zip([dt_df, mc_df, ii_df, ci_df],['dt', 'mc', 'ii', 'ci']): #[dt_df, mc_df, ii_df, ci_df],['dt', 'mc', 'ii', 'ci']
        for i, row in tqdm(df.iterrows(),total=df.shape[0]):
            skip_plot = False
            if not type(row.iloc[2]) == float:
                swc_pa = cells_pa.loc[cells_pa['cell_name']==row.iloc[2],'swc'].iloc[0]
                swc_em = cells_em.loc[cells_em['cell_name'] == row.iloc[1], 'swc'].iloc[0]
                soma_mesh_em = cells_em.loc[cells_em['cell_name'] == row.iloc[1], 'soma_mesh'].iloc[0]
                swc_clem = cells_clem.loc[cells_clem['cell_name'] == 'cell_' + row.iloc[0], 'swc'].iloc[0]
                soma_mesh_clem = cells_clem.loc[cells_clem['cell_name'] == 'cell_' + row.iloc[0], 'soma_mesh'].iloc[0]
            else:
                skip_plot=True

            if not skip_plot:
                fig,ax = plt.subplots(2,1,figsize=(12, 12))

                color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)
                projection = 'z'
                if projection == "z":
                    view = ('x', "-y")  # Set the 2D view to the X-Y plane for Z projection.
                    ylim = [-850, -50]  # Define the Y-axis limits for the Z projection.
                elif projection == 'y':
                    view = ('x', "z")  # Set the 2D view to the X-Z plane for Y projection.
                    ylim = [-30, 300]
                ylim = [-850, -50]
                ylim = [-30, 300]

                navis.plot2d(brain_meshes, color=color_meshes,
                             alpha=0.2, linewidth=0.5, method='2d', view=view, group_neurons=False,
                             rasterize=True, ax=ax[0],
                             scalebar="20 um")

                navis.plot2d(swc_pa, color='red', alpha=1, linewidth=0.5,
                             method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[0])

                navis.plot2d(swc_clem, color='black', alpha=1, linewidth=0.5,
                             method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[0])
                navis.plot2d(soma_mesh_clem, color='black', alpha=1, linewidth=0.5,
                             method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[0])

                navis.plot2d(swc_em, color='cyan', alpha=1, linewidth=0.5,
                             method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[0])
                navis.plot2d(soma_mesh_em, color='cyan', alpha=1, linewidth=0.5,
                             method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[0])

                # navis.plot2d(axon_mesh_list, color=color_list, alpha=1, linewidth=0.5,
                ax[0].set_aspect('equal')
                ax[0].axvline(250, color=(0.85, 0.85, 0.85, 0.2), linestyle='--', alpha=0.5, zorder=0)
                ax[0].set_xlim(50, 350)  # Standardize the plot dimensions.
                # Set specific limits for the Y-axis based on the projection.
                ax[0].set_facecolor('white')
                ax[0].axis('off')
                #f'{cell_class}\nCLEM: {row.iloc[0]}\nEM: {row.iloc[1]}\npaGFP: {row.iloc[2]}.pdf'

                # fig.savefig(path_to_figure_dir / f'{cell_class}_clem{row.iloc[0]}_em{row.iloc[1]}pa_{row.iloc[2]}.pdf', dpi=600)
                # fig = plt.figure(figsize=(12, 12))

                color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)
                projection = 'y'
                if projection == "z":
                    view = ('x', "-y")  # Set the 2D view to the X-Y plane for Z projection.
                    ylim = [-850, -50]  # Define the Y-axis limits for the Z projection.
                elif projection == 'y':
                    view = ('x', "z")  # Set the 2D view to the X-Z plane for Y projection.
                    ylim = [-80, 400]

                navis.plot2d(brain_meshes, color=color_meshes,volume_outlines =False,
                             alpha=0.2, linewidth=0.5, method='2d', view=view, group_neurons=False,
                             rasterize=True, ax=ax[1],
                             scalebar="20 um")

                navis.plot2d(swc_pa, color='red', alpha=1, linewidth=0.5,
                             method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[1])

                navis.plot2d(swc_clem, color='black', alpha=1, linewidth=0.5,
                             method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[1])
                navis.plot2d(soma_mesh_clem, color='black', alpha=1, linewidth=0.5,
                             method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[1])

                navis.plot2d(swc_em, color='cyan', alpha=1, linewidth=0.5,
                             method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[1])
                navis.plot2d(soma_mesh_em, color='cyan', alpha=1, linewidth=0.5,
                             method='2d', view=view, group_neurons=True, rasterize=False, ax=ax[1])

                ax[1].set_aspect('equal')
                ax[1].axvline(250, color=(0.85, 0.85, 0.85, 0.2), linestyle='--', alpha=0.5, zorder=0)
                ax[1].set_xlim(50, 350)  # Standardize the plot dimensions.
                # Set specific limits for the Y-axis based on the projection.
                ax[1].set_facecolor('white')
                ax[1].axis('off')
                #{cell_class}\nCLEM: {row.iloc[0]}\nEM: {row.iloc[1]}\npaGFP: {row.iloc[2]}
                fig.savefig(path_to_figure_dir / f'{cell_class}_clem{row.iloc[0]}_em{row.iloc[1]}pa_{row.iloc[2]}.pdf', dpi=300)
                fig.savefig(path_to_figure_dir / f'{cell_class}_clem{row.iloc[0]}_em{row.iloc[1]}pa_{row.iloc[2]}.png', dpi=300)