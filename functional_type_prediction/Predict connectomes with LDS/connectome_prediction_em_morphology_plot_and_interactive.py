from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_predict_jon_cells import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.visualization.make_figures_FK import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.find_branches import *
import numpy as np
import plotly
# matplotlib.use('TkAgg')
from tqdm import tqdm
import winsound

def extract_prediction(result):
    cleaned = [x.replace('"', "") for x in result.split(' ')[2:]]
    a_proba = np.array([eval(x[3:]) for x in cleaned])
    a_label = np.array([x[:2] for x in cleaned])
    prediced_label = a_label[np.argmax(a_proba)]

    return prediced_label
if __name__ == '__main__':

    #set variables
    np.set_printoptions(suppress=True)
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')
    brain_meshes = load_brs(path_to_data, 'raphe')
    width_brain = 495.56
    path_to_figure_dir = path_to_data / 'make_figures_FK_output' / 'EmG_connectome_prediction'
    os.makedirs(path_to_figure_dir,exist_ok=True)

    #load cells

    cells_em = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['em'], load_repaired=True,load_both=True,mirror=False)
    cells_em = cells_em.reset_index(drop=True)



    # for presynaptic_cell in [x for x in cells_em.presynaptic.unique() if x != 'seed_cell']:
    #     temp_connectome_df = cells_em.loc[(cells_em['presynaptic'] == presynaptic_cell) | (cells_em['cell_name'] == presynaptic_cell),:]
    #     color_list = []
    #     swc_list = []
    #     alpha_list = []
    #     for i,cell in temp_connectome_df.iterrows():
    #
    #         add_cell = True
    #         if cell['presynaptic']!='seed_cell':
    #
    #             for tested_feature in cell.index[22:]:
    #                 if not cell[tested_feature] and 'reduced' not in tested_feature:
    #                     add_cell = False
    #         if add_cell:
    #             color_list.append(color_dict[cell['prediction']])
    #             swc_list.append(cell['swc'])
    #
    #
    #     fig = navis.plot3d(swc_list, backend='plotly',colors=color_list,
    #                        width=1920, height=1080, hover_name=True)
    #     fig = navis.plot3d(brain_meshes, backend='plotly', fig=fig,
    #                        width=1920, height=1080, hover_name=True)
    #     fig.update_layout(
    #         scene={
    #             'xaxis': {'autorange': 'reversed'},  # reverse !!!
    #             'yaxis': {'autorange': True},
    #
    #             'zaxis': {'autorange': True},
    #             'aspectmode': "data",
    #             'aspectratio': {"x": 1, "y": 1, "z": 1}},
    #         title = dict(text=f'connectome not reduced seed cell = {presynaptic_cell}', font=dict(size=20), automargin=True, yref='paper')
    #     )
    #     # os.makedirs(path_to_data / 'make_figures_FK_output'/'LDS_NBLAST_predictions_reduced_features_clem',exist_ok=True)
    #     # temp_file_name= path_to_data / 'make_figures_FK_output'/'LDS_NBLAST_predictions_reduced_features_clem'/f"{acronym_dict[y_pred]}_{cell_name}.html"
    #     plotly.offline.plot(fig, filename='test.html', auto_open=True, auto_play=False)
    #
    #
    #
    # for presynaptic_cell in [x for x in cells_em.presynaptic.unique() if x != 'seed_cell']:
    #     temp_connectome_df = cells_em.loc[(cells_em['presynaptic'] == presynaptic_cell) | (cells_em['cell_name'] == presynaptic_cell),:]
    #     color_list = []
    #     swc_list = []
    #     alpha_list = []
    #     for i,cell in temp_connectome_df.iterrows():
    #
    #         add_cell = True
    #         if cell['presynaptic']!='seed_cell':
    #
    #             for tested_feature in cell.index[22:]:
    #                 if not cell[tested_feature] and 'reduced' in tested_feature:
    #                     add_cell = False
    #         if add_cell:
    #             color_list.append(color_dict[cell['prediction_reduced']])
    #             swc_list.append(cell['swc'])
    #
    #
    #     fig = navis.plot3d(swc_list, backend='plotly',colors=color_list,
    #                        width=1920, height=1080, hover_name=True)
    #     fig = navis.plot3d(brain_meshes, backend='plotly', fig=fig,
    #                        width=1920, height=1080, hover_name=True)
    #     fig.update_layout(
    #         scene={
    #             'xaxis': {'autorange': 'reversed'},  # reverse !!!
    #             'yaxis': {'autorange': True},
    #
    #             'zaxis': {'autorange': True},
    #             'aspectmode': "data",
    #             'aspectratio': {"x": 1, "y": 1, "z": 1}},
    #         title = dict(text=f'connectome reduced seed cell = {presynaptic_cell}', font=dict(size=20), automargin=True, yref='paper')
    #     )
    #     # os.makedirs(path_to_data / 'make_figures_FK_output'/'LDS_NBLAST_predictions_reduced_features_clem',exist_ok=True)
    #     # temp_file_name= path_to_data / 'make_figures_FK_output'/'LDS_NBLAST_predictions_reduced_features_clem'/f"{acronym_dict[y_pred]}_{cell_name}.html"
    #     plotly.offline.plot(fig, filename='test.html', auto_open=True, auto_play=False)

    for i,cell in cells_em.iterrows():
        temp_meta_path = cell['metadata_path']
        temp_meta_path = Path(str(temp_meta_path)[:-4] + '_with_prediction.txt')

        with open(temp_meta_path, 'r') as f:
            t = f.read()

        meta_data_list = [x for x in t.split('\n') if x != ""]

        cells_em.loc[i,'prediction'] = extract_prediction(meta_data_list[12])
        cells_em.loc[i, 'all_proba'] = meta_data_list[12]
        temp_predictions = [x.strip('"') for x in meta_data_list[12].split(' ')[2:]]
        cells_em.loc[i, 'proba_DT'] = eval(temp_predictions[0].split(':')[1])
        cells_em.loc[i, 'proba_CI'] = eval(temp_predictions[1].split(':')[1])
        cells_em.loc[i, 'proba_II'] = eval(temp_predictions[2].split(':')[1])
        cells_em.loc[i, 'proba_MC'] = eval(temp_predictions[3].split(':')[1])
        cells_em.loc[i, 'prediction_reduced'] = extract_prediction(meta_data_list[13])
        cells_em.loc[i, 'nblast_test'] = eval(meta_data_list[14].split(' ')[2])
        cells_em.loc[i, 'nblast_test_specific'] = eval(meta_data_list[15].split(' ')[2])
        cells_em.loc[i, 'proba_test'] = eval(meta_data_list[16].split(' ')[2])
        cells_em.loc[i, 'proba_test_reduced'] = eval(meta_data_list[17].split(' ')[2])
        cells_em.loc[i, 'nov_OCSVM'] = eval(meta_data_list[18].split(' ')[2])
        cells_em.loc[i, 'nov_OCSVM_reduced'] = eval(meta_data_list[19].split(' ')[2])
        cells_em.loc[i, 'nov_IF'] = eval(meta_data_list[20].split(' ')[2])
        cells_em.loc[i, 'nov_IF_reduced'] = eval(meta_data_list[21].split(' ')[2])
        cells_em.loc[i, 'nov_LOF'] = eval(meta_data_list[22].split(' ')[2])
        cells_em.loc[i, 'nov_LOF_reduced'] = eval(meta_data_list[23].split(' ')[2])

    #connectomeS
    def color_weakener(hex_color,factor=0.75):
        """
        Weakens the color by adjusting its alpha (transparency) value.
        Parameters:
        hex_color (str): The hexadecimal color code with alpha (e.g., '#feb326b3').
        factor (float): The factor by which to adjust the alpha value (default is 0.75).
        Returns:
        str: The new hexadecimal color code with adjusted alpha.
        """
        # Remove the hash symbol if present
        hex_color = hex_color.lstrip('#')
        # Split the hex color into RGB and alpha components
        rgb = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
        alpha = int(hex_color[6:8], 16) if len(hex_color) == 8 else 255
        # Adjust the alpha value
        new_alpha = int(alpha * factor)
        # Construct the new hex color with adjusted alpha
        output_hex = '#{:02x}{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2], new_alpha)
        return output_hex


    color_dict = {
        "II": '#feb326b3',
        "CI": '#e84d8ab3',
        "DT": '#64c5ebb3',
        "MC": '#7f58afb3',
    }

    color_dict_rgba = {
        "II": [254,179,38,1],
        "CI": [232,77,138,1],
        "DT": [100,197,235,1],
        "MC": [127,88,175,1],
    }



    for presynaptic_cell in tqdm(x for x in cells_em.presynaptic.unique() if x != 'seed_cell'):
        temp_connectome_df = cells_em.loc[(cells_em['presynaptic'] == presynaptic_cell) | (cells_em['cell_name'] == presynaptic_cell),:]
        color_list = []
        swc_list = []
        soma_mesh_list = []
        axon_mesh_list = []
        dendrite_mesh_list = []
        alpha_list = []
        legend_dict = {}
        for i,cell in temp_connectome_df.iterrows():

            add_cell = True
            if cell['presynaptic']!='seed_cell':


                if not cell['nblast_test']:
                    add_cell = False
            if add_cell:
                color_list.append(color_dict[cell['prediction_reduced']])

                swc_list.append(cell['swc'])
                swc_list[-1].name = "_".join(swc_list[-1].name.split('_')[:3])


                soma_mesh_list.append(navis.simplify_mesh(cell['soma_mesh'],1000))
                soma_mesh_list[-1].name = swc_list[-1].name

                dendrite_mesh_list.append(cell['dendrite_mesh'])
                dendrite_mesh_list[-1].name = swc_list[-1].name

                axon_mesh_list.append(cell['axon_mesh'])
                axon_mesh_list[-1].name = swc_list[-1].name


        fig = navis.plot3d(swc_list, backend='plotly',colors=color_list,
                           width=1920, height=1080, hover_name=True)
        # fig = navis.plot3d(axon_mesh_list, backend='plotly', colors=color_list,
        #                    width=1920, height=1080, hover_name=True)
        # fig = navis.plot3d(dendrite_mesh_list, backend='plotly', colors=color_list, fig=fig,
        #                    width=1920, height=1080, hover_name=True)
        fig = navis.plot3d(soma_mesh_list, backend='plotly', colors=color_list,fig=fig,
                           width=1920, height=1080, hover_name=True)
        fig = navis.plot3d(brain_meshes, backend='plotly', fig=fig,
                           width=1920, height=1080, hover_name=True)
        fig.update_layout(
            scene={
                'xaxis': {'autorange': 'reversed'},  # reverse !!!
                'yaxis': {'autorange': True},

                'zaxis': {'autorange': True},
                'aspectmode': "data",
                'aspectratio': {"x": 1, "y": 1, "z": 1}},
            title = dict(text=f'connectome only_nblast_throw seed cell = {presynaptic_cell}', font=dict(size=20), automargin=True, yref='paper')
        )
        # os.makedirs(path_to_data / 'make_figures_FK_output'/'LDS_NBLAST_predictions_reduced_features_clem',exist_ok=True)
        # temp_file_name= path_to_data / 'make_figures_FK_output'/'LDS_NBLAST_predictions_reduced_features_clem'/f"{acronym_dict[y_pred]}_{cell_name}.html"
        plotly.offline.plot(fig, filename=str(path_to_figure_dir / f'{presynaptic_cell}_predicted_connectome_only_nlast_throw.html'), auto_open=True, auto_play=False)




        #2d plot
        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
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
                     rasterize=True, ax=ax,
                         scalebar="20 um")



        navis.plot2d(swc_list, color=color_list, alpha=1, linewidth=0.5,
                         method='2d', view=view, group_neurons=True, rasterize=False, ax=ax)
        navis.plot2d(soma_mesh_list, color=color_list, alpha=1, linewidth=0.5,
                         method='2d', view=view, group_neurons=True, rasterize=True, ax=ax)



        # navis.plot2d(axon_mesh_list, color=color_list, alpha=1, linewidth=0.5,
        #              method='2d', view=view, group_neurons=True, rasterize=False, ax=ax,
        #              scalebar="20 um")
        # navis.plot2d(dendrite_mesh_list, color=color_list, alpha=1, linewidth=0.5,
        #              method='2d', view=view, group_neurons=True, rasterize=False, ax=ax,
        #              scalebar="20 um")
        # navis.plot2d(swc_list, color=color_list, alpha=1, linewidth=0.5,
        #              method='2d', view=view, group_neurons=True, rasterize=False, ax=ax,
        #              scalebar="20 um")
        #
        # navis.plot2d(soma_mesh_list, color=color_list, alpha=1, linewidth=0.5,
        #              method='2d', view=view, group_neurons=True, rasterize=False, ax=ax,
        #              scalebar="20 um")

        ax.set_aspect('equal')
        ax.axvline(250, color=(0.85, 0.85, 0.85, 0.2), linestyle='--', alpha=0.5, zorder=0)
        plt.xlim(50, 300)  # Standardize the plot dimensions.
        # Set specific limits for the Y-axis based on the projection.
        ax.set_facecolor('white')
        ax.axis('off')
        fig.savefig(path_to_figure_dir / f'{presynaptic_cell}_predicted_connectome_z_projection.pdf', dpi=600)


        fig = plt.figure(figsize=(12, 12))
        ax = plt.gca()
        color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)
        projection = 'y'
        if projection == "z":
            view = ('x', "-y")  # Set the 2D view to the X-Y plane for Z projection.
            ylim = [-850, -50]  # Define the Y-axis limits for the Z projection.
        elif projection == 'y':
            view = ('x', "z")  # Set the 2D view to the X-Z plane for Y projection.
            ylim = [-80, 400]


        navis.plot2d(brain_meshes, color=color_meshes,
                     alpha=0.2, linewidth=0.5, method='2d', view=view, group_neurons=True,
                     rasterize=True, ax=ax,scalebar="20 um")

        navis.plot2d(swc_list, color=color_list, alpha=1, linewidth=0.5,
                         method='2d', view=view, group_neurons=True, rasterize=False, ax=ax)
        navis.plot2d(soma_mesh_list, color=color_list, alpha=1, linewidth=0.5,
                         method='2d', view=view, group_neurons=True, rasterize=True, ax=ax)


        # navis.plot2d(swc_list, color=color_list, alpha=1, linewidth=0.5,
        #              method='2d', view=view, group_neurons=True, rasterize=False, ax=ax,
        #              scalebar="20 um")
        #
        # navis.plot2d(soma_mesh_list, color=color_list, alpha=1, linewidth=0.5,
        #              method='2d', view=view, group_neurons=True, rasterize=False, ax=ax,
        #              scalebar="20 um")

        # navis.plot2d(axon_mesh_list, color=color_list, alpha=1, linewidth=0.5,
        #              method='2d', view=view, group_neurons=True, rasterize=False, ax=ax,
        #              scalebar="20 um")
        # navis.plot2d(dendrite_mesh_list, color=color_list, alpha=1, linewidth=0.5,
        #              method='2d', view=view, group_neurons=True, rasterize=False, ax=ax,
        #              scalebar="20 um")




        ax.set_aspect('equal')
        ax.axvline(250, color=(0.85, 0.85, 0.85, 0.2), linestyle='--', alpha=0.5, zorder=0)
        plt.xlim(50, 300)  # Standardize the plot dimensions.
        # Set specific limits for the Y-axis based on the projection.
        ax.set_facecolor('white')
        ax.axis('off')
        fig.savefig(path_to_figure_dir / f'{presynaptic_cell}_predicted_connectome_y_projection.pdf', dpi=600)

    winsound.Beep(400, 150)