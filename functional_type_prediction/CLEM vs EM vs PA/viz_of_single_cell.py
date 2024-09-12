from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_predict_jon_cells import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.visualization.make_figures_FK import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.find_branches import *
import numpy as np
import plotly

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')  # Ensure this path is set in path_configuration.txt
    brain_meshes = load_brs(path_to_data, 'raphe')
    width_brain = 495.56
    path_to_figure_dir = path_to_data / 'make_figures_FK_output' / 'find_match_cell_EM_CLEM_paGFP'
    os.makedirs(path_to_figure_dir,exist_ok=True)

    cells_clem = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem'], load_repaired=True, load_both=True,
                                             mirror=True)
    tc = cells_clem.loc[cells_clem['cell_name'] == 'cell_576460752731091853', :]
    tc = tc.iloc[0]

    tc.swc = navis.prune_twigs(tc.swc, 20, recursive=True)
    tc.swc = navis.smooth_skeleton(tc.swc, 15)
    tc.swc = navis.cut_skeleton(tc['swc'], 9112)[1]
    tc.swc = navis.prune_twigs(tc.swc, 20, recursive=True)

    aaa = tc['swc'].nodes
    tc.swc = tc['swc'].resample('10um')
    tc.swc.soma = 0
    fig = navis.plot3d(tc['swc'], backend='plotly',  colors='cyan', lw=4, width=2000, height=2000, hover_name=True)
    fig = navis.plot3d(brain_meshes, backend='plotly',width=2000, height=2000,fig=fig, hover_name=True)



    fig.update_layout(
        scene={
            'xaxis': {'showgrid': True, 'showline': True, 'gridcolor': 'black'},  # reverse !!!
            'yaxis': {'autorange': True, 'showgrid': True, 'showline': False, 'gridcolor': 'black'},

            'zaxis': {'autorange': True, 'showgrid': True, 'gridcolor': 'black', 'showline': False},
            'aspectmode': "data",
            'aspectratio': {"x": 1, "y": 1, "z": 1},
            'camera': {'projection': {'type': 'orthographic'},
                       'eye':{"x":0,"y":0,"z":1},
                       'center':{"x":0,"y":0,"z":0}},}

    )
    plotly.offline.plot(fig, filename=r"C:\Users\ag-bahl\Downloads\test.html", auto_open=True, auto_play=False)




