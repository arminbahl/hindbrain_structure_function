from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_predict_jon_cells import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells2df import *
from hindbrain_structure_function.functional_type_prediction.classifier_prediction.LDS_single_cell_prediction import *
from hindbrain_structure_function.visualization.make_figures_FK import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.find_branches import *
import numpy as np
import plotly


# matplotlib.use('TkAgg')

def extract_prediction(result):
    cleaned = [x.replace('"', "") for x in result.split(' ')[2:]]
    a_proba = np.array([eval(x[3:]) for x in cleaned])
    a_label = np.array([x[:2] for x in cleaned])
    prediced_label = a_label[np.argmax(a_proba)]

    return prediced_label
if __name__ == '__main__':

    #set variables
    np.set_printoptions(suppress=True)
    path_to_data = Path('C:/Users/ag-bahl/Desktop/hindbrain_structure_function/nextcloud_folder/CLEM_paper_data')  # Ensure this path is set in path_configuration.txt
    brain_meshes = load_brs(path_to_data, 'raphe')
    width_brain = 495.56
    path_to_figure_dir = path_to_data / 'make_figures_FK_output' / 'find_match_cell_EM_CLEM_paGFP'
    os.makedirs(path_to_figure_dir,exist_ok=True)


    dict2abrv = {'integrator ipsilateral':'II','integrator contralateral':'CI','motor command':'MC',"dynamic threshold":'DT'}

    #load cells

    cells_em = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['em'], load_repaired=True,load_both=True,mirror=True)
    cells_em = cells_em.reset_index(drop=True)
    cells_em = cells_em.sort_values('cell_name')

    cells_clem = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['clem'], load_repaired=True, load_both=True,
                                             mirror=True)
    cells_clem = cells_clem.loc[(cells_clem['function'] != 'nan') | (cells_clem['function'].isna()), :]
    cells_clem = cells_clem.reset_index(drop=True)
    cells_clem = cells_clem.sort_values('cell_name')


    cells_clem['class'] = cells_clem.apply(lambda x: dict2abrv[x.function.replace('_', ' ')] if x.function != 'integrator' else dict2abrv[x.function + " " + x.morphology], axis=1)

    cells_pa = load_cells_predictor_pipeline(path_to_data=Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data'), modalities=['pa'], load_repaired=False, load_both=True,
                                             mirror=True)
    cells_pa = cells_pa.reset_index(drop=True)

    cells_pa['class'] = cells_pa.apply(lambda x: dict2abrv[x.function.replace('_',' ')] if x.function != 'integrator' else dict2abrv[x.function +" "+ x.morphology],axis=1 )
    cells_pa = cells_pa.sort_values('cell_name')

    for i,cell in cells_em.iterrows():
        temp_meta_path = cell['metadata_path']
        temp_meta_path = Path(str(temp_meta_path)[:-4] + '_with_prediction.txt')

        with open(temp_meta_path, 'r') as f:
            t = f.read()

        meta_data_list = [x for x in t.split('\n') if x != ""]

        cells_em.loc[i,'prediction'] = extract_prediction(meta_data_list[12])
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

    for unique_cell_class in cells_em.prediction.unique():
        subset_clem = cells_clem.loc[cells_clem['class'] == unique_cell_class,:]
        subset_pa = cells_pa.loc[cells_pa['class'] == unique_cell_class,:]
        subset_em = cells_em.loc[cells_em['prediction'] == unique_cell_class,:]


        fig = navis.plot3d(brain_meshes, backend='plotly',
                           width=1920, height=1080, hover_name=True)
        fig = navis.plot3d(list(subset_clem['swc']), backend='plotly', fig=fig,colors='black',
                           width=1920, height=1080, hover_name=True)
        fig = navis.plot3d(list(subset_pa['swc']), backend='plotly', fig=fig,colors='red',
                           width=1920, height=1080, hover_name=True)
        fig = navis.plot3d(list(subset_em['swc']), backend='plotly', fig=fig,colors='cyan',
                           width=1920, height=1080, hover_name=True)

        fig.update_layout(
            scene={
                'xaxis': {'autorange': 'reversed'},  # reverse !!!
                'yaxis': {'autorange': True},

                'zaxis': {'autorange': True},
                'aspectmode': "data",
                'aspectratio': {"x": 1, "y": 1, "z": 1}},
            title = dict(text=f'{unique_cell_class}', font=dict(size=20), automargin=True, yref='paper')
        )

        plotly.offline.plot(fig, filename=str(path_to_figure_dir / f'{unique_cell_class}.html'), auto_open=True, auto_play=False)
