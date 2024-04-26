import navis
import numpy
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os
import plotly
from analysis_helpers.analysis.personal_dirs.Florian.clem_paper.create_metadata_swc import fix_duplicates
import h5py
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

import numpy as np
import navis
from pathlib import Path
import pandas as pd
from hindbrain_structure_function.visualization.FK_tools.load_pa_table import *
from hindbrain_structure_function.visualization.FK_tools.load_clem_table import *
from hindbrain_structure_function.visualization.FK_tools.load_mesh import *
from hindbrain_structure_function.visualization.FK_tools.load_brs import *
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
from datetime import datetime
import plotly
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    #settings
    modalities = ['clem',"pa"]



    name_time = datetime.now()

    # path settings
    path_to_data = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data')

    # load pa  table
    if 'pa' in modalities:
        pa_table = load_pa_table(path_to_data.joinpath("paGFP").joinpath("photoactivation_cells_table.csv"))
    # load clem table
    if 'clem' in modalities:
        clem_table = load_clem_table(path_to_data.joinpath('clem_zfish1').joinpath('all_cells'))

    # TODO here the loading of gregor has to go


    # concat tables
    if len(modalities) > 1:
        all_cells = pd.concat([eval(x + '_table') for x in modalities])
    elif len(modalities) == 1:
        all_cells = eval(modalities[0] + "_table")
    all_cells = all_cells.reset_index(drop=True)

    # subset dataset for keywords
    # for keyword in keywords:
    #     subset_for_keyword = all_cells['cell_type_labels'].apply(lambda current_label: True if keyword.replace("_", " ") in current_label else False)
    #     all_cells = all_cells[subset_for_keyword]

    all_cells['swc'] = np.nan
    all_cells['swc'] = all_cells['swc'].astype(object)


    # set imaging modality to clem if jon scored it
    all_cells.loc[all_cells['tracer_names'] == 'Jonathan Boulanger-Weill', 'imaging_modality'] = 'clem'  # TODO ask jonathan if we can write clem as imaging modality

    # load the meshes for each cell that fits queries in selected modalities
    for i, cell in all_cells.iterrows():
        all_cells.loc[i, :] = load_mesh(cell, path_to_data,swc=True)

    #subset to loaded swcs
    all_cells = all_cells[all_cells['swc'].apply(lambda x: True if type(x) == navis.core.skeleton.TreeNeuron else False)]
    #resample neurons
    all_cells['swc'] = all_cells['swc'].apply(lambda x: navis.resample_skeleton(x,0.1))

    from copy import deepcopy
    #create dict with smoothed versions
    smoothed_dict = {}
    smoothed_dict[0]=all_cells
    for smoothing_level in np.arange(5,100,5):
        smoothed_dict[smoothing_level] = deepcopy(all_cells)
        smoothed_dict[smoothing_level] = smoothed_dict[smoothing_level].apply(lambda x: navis.smooth_skeleton(x['swc'], window=smoothing_level) if x['imaging_modality'] == 'photoactivation' else x['swc'], axis=1)




    #nblast to find most similar neurons



    my_neuron_list = navis.NeuronList(all_cells.swc)
    dps = navis.make_dotprops(my_neuron_list, k=5, resample=False)
    nbl = navis.nblast(dps,dps, progress=False)
    nbl.index = all_cells.cell_name
    nbl.columns = all_cells.cell_name

    nbl_array = np.array(nbl)


    nbl_without_self = nbl.iloc[:(all_cells['imaging_modality']=="clem").sum(),(all_cells['imaging_modality']=="clem").sum():]

    #max index
    nbl_without_self.idxmax()
    indexer = nbl_without_self.max().sort_values(ascending=False).index

    nbl_without_self.idxmax()[indexer]

    clem_cell = all_cells[all_cells['cell_name'] == nbl_without_self.idxmax()[indexer][0]]
    pa_cell = all_cells[all_cells['cell_name']==indexer[0]]


    list_values = []
    for smoothing_level in np.arange(5,100,5):


        my_neuron_list = navis.NeuronList([navis.smooth_skeleton(pa_cell['swc'].values[0],window=smoothing_level),clem_cell['swc'].values[0]])
        dps = navis.make_dotprops(my_neuron_list, k=5, resample=False)
        nbl = navis.nblast(dps, dps, progress=False)


        nbl_array = np.array(nbl)
        list_values.append(nbl_array[1,0])

    fig = navis.plot3d([navis.smooth_skeleton(pa_cell['swc'].values[0],window=smoothing_level),clem_cell['swc'].values[0]], backend='plotly',
                        width=1920, height=1080)
    fig.update_layout(
        scene={
            'xaxis': {'autorange': True},  # reverse !!!
            'yaxis': {'autorange': True},

            'zaxis': {'autorange': True},
            'aspectmode': "data",
            'aspectratio': {"x": 1, "y": 1, "z": 1}
        }
    )

    os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("html"), exist_ok=True)

    plotly.offline.plot(fig, filename=str(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("html").joinpath("test.html")), auto_open=True,
                        auto_play=False)

