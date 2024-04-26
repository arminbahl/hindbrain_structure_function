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
import copy
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
    modalities = ["pa"]
    keywords = ['ipsilateral','integrator']
    do_mirror = False

    name_time = datetime.now()

    # path settings
    path_to_data = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data')
    os.makedirs(path_to_data.joinpath("make_figures_FK_output").joinpath('nblast_experiments'),exist_ok=True)
    path_to_output_folder = path_to_data.joinpath("make_figures_FK_output").joinpath('nblast_experiments')


    # load pa  table
    if 'pa' in modalities:
        pa_table = load_pa_table(path_to_data.joinpath("paGFP").joinpath("photoactivation_cells_table.csv"))



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
    all_cells['swc_smooth'] = np.nan
    all_cells['swc_smooth'] = all_cells['swc_smooth'].astype(object)


    # load the meshes for each cell that fits queries in selected modalities
    for i, cell in all_cells.iterrows():
        all_cells.loc[i, :] = load_mesh(cell, path_to_data,swc=True,use_smooth_pa=False)

        all_cells.loc[i, 'swc'].soma = 1
        all_cells.loc[i, 'swc'].soma_radius = 2
        if cell['imaging_modality'] == 'photoactivation':

            all_cells.loc[i, 'swc_smooth'] = navis.smooth_skeleton(all_cells.loc[i, 'swc'],window=10)


    #subset to loaded swcs
    all_cells = all_cells[all_cells['swc'].apply(lambda x: True if type(x) == navis.core.skeleton.TreeNeuron else False)]

    #mirror the pa neurons
    midline_loc = 495.56/2

    if do_mirror:
        for i,cell in all_cells.iterrows():
            temp_cell = copy.deepcopy(cell)
            temp_navis_object = copy.deepcopy(cell['swc'])
            temp_navis_object.nodes.loc[:,'x'] = temp_navis_object.nodes.apply(lambda x: midline_loc-(x['x']-midline_loc),axis=1)
            temp_navis_object.name = temp_cell.cell_name + "_mirror"

            temp_cell['cell_name'] = temp_cell.cell_name + "_mirror"
            temp_cell['swc'] = temp_navis_object
            all_cells = pd.concat([all_cells,pd.DataFrame([temp_cell])],ignore_index=True,axis=0)



    # load gregors cells
    # data seed cells
    path_seed_cells = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\em_zfish1\data_seed_cells\output_data')
    path_seed_cells = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\gregor_seed_dropbbox')
    path_postsynaptic = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\em_zfish1\data_cell_89189_postsynaptic_partners\output_data')

    em_cells = []
    seed_or_post = []

    for cell in os.listdir(path_seed_cells):
        temp_df = pd.DataFrame(columns=all_cells.columns,index=[0])
        temp_df['swc'] = np.nan
        temp_df['swc'] = temp_df['swc'].astype(object)

        path_to_cell = path_seed_cells.joinpath(cell).joinpath('mapped').joinpath(rf"em_fish1_{cell.split('_')[-1]}_mapped.swc")
        if path_to_cell.exists():
            temp_df['cell_name'] = rf"em_fish1_{cell.split('_')[-1]}"
            temp_df['swc'] = navis.read_swc(path_to_cell, units="um")
            temp_df['imaging_modality'] = 'em'
            all_cells = pd.concat([all_cells, temp_df])
        else:
            print(cell, "not mapped")

    for cell in os.listdir(path_postsynaptic):
        temp_df = pd.DataFrame(columns=all_cells.columns,index=[0])
        temp_df['swc'] = np.nan
        temp_df['swc'] = temp_df['swc'].astype(object)
        path_to_cell = path_postsynaptic.joinpath(cell).joinpath('mapped').joinpath(rf"em_fish1_{cell.split('_')[-1]}_mapped.swc")
        if path_to_cell.exists():
            temp_df['cell_name'] = rf"em_fish1_{cell.split('_')[-1]}"
            temp_df['swc'] = navis.read_swc(path_to_cell, units="um")
            temp_df['imaging_modality'] = 'em'
            all_cells = pd.concat([all_cells, temp_df])

        else:
            print(cell, "not mapped")

    all_cells.loc[all_cells['swc_smooth'].isna(),"swc_smooth"] = all_cells.loc[all_cells['swc_smooth'].isna(),"swc"]

    #nblast part
    all_cells['swc'] = all_cells['swc'].apply(lambda x: navis.resample_skeleton(x, 0.1))

    #nblast


    my_neuron_list = navis.NeuronList(all_cells[all_cells.cell_name.apply(lambda x: False if "mirror" in x else True)].swc)

    dps = navis.make_dotprops(my_neuron_list, k=5, resample=False)
    nbl = navis.nblast(dps,dps, progress=False)
    nbl_array = np.array(nbl)


    #hirachical clustering

    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette, fcluster

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcl
    import seaborn as sns

    nbl_mean = (nbl + nbl.T) / 2
    nbl_dist = 1 - nbl_mean
    nbl_vec = squareform(nbl_dist, checks=False)
    Z = linkage(nbl_vec, method='ward')
    assigned_cluster = fcluster(Z, 20, criterion="maxclust")
    all_cells['hierarchical_cluster'] = assigned_cluster
    all_cells_sorted= all_cells.sort_values('hierarchical_cluster')

    # em_neuron_list = navis.NeuronList(all_cells.loc[all_cells['imaging_modality']=="em","swc"])
    # dps_em = navis.make_dotprops(em_neuron_list, k=5, resample=False)
    # nbl_em = navis.nblast(dps_em, dps_em, progress=False)
    #
    # pa_neuron_list = navis.NeuronList(all_cells.loc[all_cells['imaging_modality'] == "photoactivation", "swc"])
    # dps_pa = navis.make_dotprops(pa_neuron_list, k=5, resample=False)
    # nbl_pa = navis.nblast(dps_pa, dps_pa, progress=False)




    if do_mirror:
        my_neuron_list_mirror = navis.NeuronList(all_cells[all_cells.cell_name.apply(lambda x: True if "mirror" in x else False)].swc)
        dps_mirror = navis.make_dotprops(my_neuron_list_mirror, k=5, resample=False)
        nbl_mirror = navis.nblast(dps_mirror,dps, progress=False)


    nbl.index = all_cells[all_cells.cell_name.apply(lambda x: False if "mirror" in x else True)].cell_name
    nbl.columns = all_cells[all_cells.cell_name.apply(lambda x: False if "mirror" in x else True)].cell_name
    if do_mirror:
        nbl_mirror.index = all_cells[all_cells.cell_name.apply(lambda x: True if "mirror" in x else False)].cell_name
        nbl_mirror.columns = all_cells[all_cells.cell_name.apply(lambda x: False if "mirror" in x else True)].cell_name





    #cut it correctly so only one dataframe
    temp_df = all_cells[all_cells.cell_name.apply(lambda x: False if "mirror" in x else True)]
    nbl = nbl.loc[list(temp_df['imaging_modality'] == 'em'),list(temp_df['imaging_modality'] != 'em')]
    if do_mirror:
        nbl_mirror = nbl_mirror.T.loc[list(temp_df['imaging_modality'] == 'em'), :]
        nbl_all = pd.concat([nbl, nbl_mirror],axis=1)
        nbl_all = nbl_all.sort_values(axis=0,by=list(nbl_all.max().sort_values(ascending=False).index),ascending=False)
    else:
        nbl_all = nbl
    #plot as heatmap
    sort_em = all_cells_sorted.loc[all_cells_sorted['imaging_modality']=='em',"cell_name"]
    sort_pa = all_cells_sorted.loc[all_cells_sorted['imaging_modality'] == 'photoactivation', "cell_name"]
    nbl_array = np.array(nbl_all.loc[sort_em,sort_pa])

    plt.figure(dpi=500)
    plt.imshow(nbl_array,cmap="PiYG")
    plt.xticks(range(len(nbl_all.columns)), nbl_all.columns,rotation=90)
    plt.yticks(range(len(nbl_all.index)),nbl_all.index,rotation=0)
    plt.gca().tick_params(axis='both', which='major', labelsize=3)
    plt.colorbar()
    plt.show()


    #display
    all_cells = all_cells.reset_index(drop=True)


    brain_meshes = load_brs(path_to_data, load_FK_regions=True)
    selected_meshes = ["Retina", 'Midbrain', "Forebrain", "Habenula", "Hindbrain", "Spinal Cord"]
    brain_meshes = [mesh for mesh in brain_meshes if mesh.name in selected_meshes]
    color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)

    list_cells = list(all_cells.loc[all_cells['imaging_modality']=='em','swc'])
    list_cells = list(all_cells['swc'])

    fig = navis.plot3d(list_cells+brain_meshes, backend='plotly',
                        width=1920, height=1080)
    fig.update_layout(
        scene={
            'xaxis': {'autorange': 'reversed'},  # reverse !!!
            'yaxis': {'autorange': True},

            'zaxis': {'autorange': True},
            'aspectmode': "data",
            'aspectratio': {"x": 1, "y": 1, "z": 1}
        }
    )



    plotly.offline.plot(fig, filename=str(path_to_output_folder.joinpath('all_cell_nblasted.html')), auto_open=False, auto_play=False)

    os.makedirs(path_to_output_folder.joinpath('table_outputs'),exist_ok=True)
    path_table_output = path_to_output_folder.joinpath('table_outputs')
    nbl_all.to_csv(path_table_output.joinpath('all_cells_nblast_result.csv'))
    nbl_all.to_excel(path_table_output.joinpath('all_cells_nblast_result.xlsx'))


    #look for matches

    nbl_match = nbl_all[nbl_all>=0.1].dropna(how='all',axis=0)
    nbl_match = nbl_match.dropna(how='all', axis=1)

    plt.figure(dpi=300)
    plt.imshow(nbl_match)
    plt.xticks(range(len(nbl_match.columns)), nbl_match.columns,rotation=90)
    plt.yticks(range(len(nbl_match.index)),nbl_match.index,rotation=0)
    plt.gca().tick_params(axis='both', which='major', labelsize=3)
    plt.colorbar()
    os.makedirs(path_to_output_folder.joinpath('heatmap_outputs'), exist_ok=True)
    path_heatmap_output = path_to_output_folder.joinpath('heatmap_outputs')
    plt.savefig(path_heatmap_output.joinpath(r'all_pa_cells_with_match_above0.1.png'),dpi=300)
    plt.savefig(path_heatmap_output.joinpath(r'all_pa_cells_with_match_above0.1.pdf'), dpi=300)
    plt.savefig(path_heatmap_output.joinpath(r'all_pa_cells_with_match_above0.1.svg'), dpi=300)
    plt.show()


    color_cell_type_dict = {"integrator": "red",
                            "dynamic_threshold": "cyan",
                            "motor_command": "purple", }

    nbl_match_sorted = nbl_match.loc[:,nbl_match.max().sort_values(ascending=False).index]

    os.makedirs(path_to_output_folder.joinpath('pa_cell_matches_html'), exist_ok=True)
    path_html_match_output = path_to_output_folder.joinpath('pa_cell_matches_html')

    for pa_cell in nbl_match_sorted:

        matching_cells_names = list(nbl_match.loc[:,pa_cell].dropna().index)
        matching_cells = all_cells.loc[all_cells['cell_name'].isin(matching_cells_names), :]
        target_cell = all_cells.loc[all_cells['cell_name']==pa_cell, :]
        viz_df = pd.concat([target_cell,matching_cells])

        brain_meshes = load_brs(Path(r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data"), load_FK_regions=True)
        selected_meshes = ["Retina", 'Midbrain', "Forebrain", "Habenula", "Hindbrain", "Spinal Cord"]
        brain_meshes = [mesh for mesh in brain_meshes if mesh.name in selected_meshes]

        for label in target_cell.cell_type_labels.iloc[0]:
            if label.replace("_", " ") in color_cell_type_dict.keys() or label in color_cell_type_dict.keys():
                pa_cell_color = color_cell_type_dict[label]



        colors = [pa_cell_color] + ["black"]*len(matching_cells_names)

        fig = navis.plot3d(list(viz_df['swc'])+brain_meshes,color=colors, backend='plotly',width=1920, height=1080)
        fig.update_layout(
            scene={
                'xaxis': {'autorange': 'reversed'},  # reverse !!!
                'yaxis': {'autorange': True},

                'zaxis': {'autorange': True},
                'aspectmode': "data",
                'aspectratio': {"x": 1, "y": 1, "z": 1}
            }
        )

        plotly.offline.plot(fig, filename=str(path_html_match_output.joinpath(f"{pa_cell}_nblast_matches.html")), auto_open=False,auto_play=False)


    #find a direct partner for each pa cell


    nbl_direct_partner = nbl_all.loc[(nbl_all >= 0.1).any(axis=1),:]
    nbl_direct_partner = nbl_direct_partner.loc[:, (nbl_direct_partner >= 0.1).any(axis=0)]
    all_em_cells = list(nbl_direct_partner.index)
    all_pa_cells = list(nbl_direct_partner.columns)

    all_values = np.sort(np.unique(nbl_direct_partner))[::-1]

    for nblast_value in all_values:
        pair = nbl_direct_partner.where(nbl_direct_partner==nblast_value).dropna(how='all', axis=0).dropna(how='all', axis=1)
        if not pair.empty:
            all_em_cells.remove(pair.index[0])
            all_pa_cells.remove(pair.columns[0])
            nbl_direct_partner.loc[nbl_direct_partner[pair.columns[0]]!=nblast_value,pair.columns[0]] = np.nan
            nbl_direct_partner.loc[pair.index[0],nbl_direct_partner.loc[pair.index[0],:]!=nblast_value] = np.nan

    nbl_direct_partner = nbl_direct_partner.loc[(nbl_direct_partner >= 0.1).any(axis=1), :]
    nbl_direct_partner = nbl_direct_partner.loc[:, (nbl_direct_partner >= 0.1).any(axis=0)]

    plt.figure(dpi=300)
    plt.imshow(nbl_all.loc[nbl_direct_partner.index,nbl_direct_partner.columns])


    for y in range(nbl_direct_partner.shape[0]):
        max_pa_match_name = nbl_direct_partner.iloc[y, :].idxmax()
        x = nbl_direct_partner.columns.get_loc(max_pa_match_name)
        my_sq = np.array([[x - 0.5, y - 0.5], [x + 0.5, y - 0.5], [x + 0.5, y + 0.5], [x - 0.5, y + 0.5], [x - 0.5, y - 0.5]])
        plt.plot(my_sq[:, 0], my_sq[:, 1], color='red')
        print(x, y)
        print(max_pa_match_name)

    plt.xticks(range(len(nbl_direct_partner.columns)), nbl_direct_partner.columns,rotation=90)
    plt.yticks(range(len(nbl_direct_partner.index)),nbl_direct_partner.index,rotation=0)
    plt.gca().tick_params(axis='both', which='major', labelsize=3)
    cb = plt.colorbar()
    cb.outline.set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    os.makedirs(path_to_output_folder.joinpath('heatmap_outputs'), exist_ok=True)
    path_heatmap_output = path_to_output_folder.joinpath('heatmap_outputs')
    plt.savefig(path_heatmap_output.joinpath(r'direct_partners_above0.1.png'),dpi=300)
    plt.savefig(path_heatmap_output.joinpath(r'direct_partners_above0.1.pdf'), dpi=300)
    plt.savefig(path_heatmap_output.joinpath(r'direct_partners_above0.1.svg'), dpi=300)

    plt.show()

    os.makedirs(path_to_output_folder.joinpath('single_cells'),exist_ok=True)
    os.makedirs(path_to_output_folder.joinpath('single_cells').joinpath('png'),exist_ok=True)
    os.makedirs(path_to_output_folder.joinpath('single_cells').joinpath('svg'), exist_ok=True)
    os.makedirs(path_to_output_folder.joinpath('single_cells').joinpath('pdf'), exist_ok=True)
    os.makedirs(path_to_output_folder.joinpath('single_cells').joinpath('mutual_matching_cells'), exist_ok=True)
    os.makedirs(path_to_output_folder.joinpath('single_cells').joinpath('mutual_matching_cells').joinpath('png'), exist_ok=True)
    os.makedirs(path_to_output_folder.joinpath('single_cells').joinpath('mutual_matching_cells').joinpath('svg'), exist_ok=True)
    os.makedirs(path_to_output_folder.joinpath('single_cells').joinpath('mutual_matching_cells').joinpath('pdf'), exist_ok=True)
    path_output_single_cell = path_to_output_folder.joinpath('single_cells')
    path_output_mutual_matching_cells = path_to_output_folder.joinpath('single_cells').joinpath('mutual_matching_cells')

    for pa_cell in nbl_direct_partner:

        matching_cells_names = list(nbl_direct_partner.loc[:,pa_cell].dropna().index)
        matching_cells = all_cells.loc[all_cells['cell_name'].isin(matching_cells_names), :]
        target_cell = all_cells.loc[all_cells['cell_name']==pa_cell, :]
        viz_df = pd.concat([target_cell,matching_cells])

        brain_meshes = load_brs(Path(r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data"), load_FK_regions=True)
        selected_meshes = ["Retina", 'Midbrain', "Forebrain", "Habenula", "Hindbrain", "Spinal Cord"]
        brain_meshes = [mesh for mesh in brain_meshes if mesh.name in selected_meshes]

        for label in target_cell.cell_type_labels.iloc[0]:
            if label.replace("_", " ") in color_cell_type_dict.keys() or label in color_cell_type_dict.keys():
                pa_cell_color = color_cell_type_dict[label]



        colors = [pa_cell_color] + ["black"]*len(matching_cells_names)

        fig = navis.plot3d(list(viz_df['swc_smooth'])+brain_meshes,color=colors, backend='plotly',width=1920, height=1080)
        fig.update_layout(
            scene={
                'xaxis': {'autorange': 'reversed'},  # reverse !!!
                'yaxis': {'autorange': True},

                'zaxis': {'autorange': True},
                'aspectmode': "data",
                'aspectratio': {"x": 1, "y": 1, "z": 1}
            }
        )

        plotly.offline.plot(fig, filename=str(path_html_match_output.joinpath(f"{pa_cell}_nblast_mutual_matches.html")), auto_open=False,auto_play=False)

       
        navis_element = list(viz_df['swc_smooth'])


        fig, ax = navis.plot2d(navis_element, color=colors, linewidth=0.5, method='2d', view=('x', "-y"), group_neurons=True, rasterize=True,scalebar="10 um")
        ax.axvline(250, color='gray', linestyle='--', alpha=0.5, zorder=0)
        fig.savefig(path_output_mutual_matching_cells.joinpath("png").joinpath(f"{pa_cell}_z_projection_mutual_matching.png"), dpi=300)
        fig.savefig(path_output_mutual_matching_cells.joinpath("svg").joinpath(f"{pa_cell}_z_projection_mutual_matching.svg"), dpi=300)
        fig.savefig(path_output_mutual_matching_cells.joinpath("pdf").joinpath(f"{pa_cell}_z_projection_mutual_matching.pdf"), dpi=300)
        fig, ax = navis.plot2d(navis_element, color=colors, linewidth=0.5, method='2d', view=('x', "z"), group_neurons=True, rasterize=True,scalebar="10 um")
        ax.axvline(250, color='gray', linestyle='--', alpha=0.5, zorder=0)
        fig.savefig(path_output_mutual_matching_cells.joinpath("png").joinpath(f"{pa_cell}_y_projection_mutual_matching.png"), dpi=300)
        fig.savefig(path_output_mutual_matching_cells.joinpath("svg").joinpath(f"{pa_cell}_y_projection_mutual_matching.svg"), dpi=300)
        fig.savefig(path_output_mutual_matching_cells.joinpath("pdf").joinpath(f"{pa_cell}_y_projection_mutual_matching.pdf"), dpi=300)



    os.makedirs(path_to_output_folder.joinpath('single_cells').joinpath('single_cells_mutual'), exist_ok=True)
    os.makedirs(path_to_output_folder.joinpath('single_cells').joinpath('single_cells_mutual').joinpath('png'), exist_ok=True)
    os.makedirs(path_to_output_folder.joinpath('single_cells').joinpath('single_cells_mutual').joinpath('svg'), exist_ok=True)
    os.makedirs(path_to_output_folder.joinpath('single_cells').joinpath('single_cells_mutual').joinpath('pdf'), exist_ok=True)
    path_output_single_cells_mutual = path_to_output_folder.joinpath('single_cells').joinpath('single_cells_mutual')

    for cell in np.concatenate([nbl_direct_partner.columns,nbl_direct_partner.index]):
        navis_element = all_cells.loc[all_cells['cell_name']==cell,"swc_smooth"]

        try:
            for label in all_cells.loc[all_cells['cell_name']==cell,'cell_type_labels'].iloc[0]:
                if label in color_cell_type_dict.keys():
                    color = color_cell_type_dict[label]

        except:
            color = 'k'


        fig, ax = navis.plot2d(navis_element.iloc[0], color=color, linewidth=4, method='2d', view=('x', "-y"), group_neurons=True, rasterize=True)
        fig.savefig(path_output_single_cells_mutual.joinpath("png").joinpath(f"{cell}_z_projection_mutual.png"),dpi=300)
        fig.savefig(path_output_single_cells_mutual.joinpath("svg").joinpath(f"{cell}_z_projection_mutual.svg"), dpi=300)
        fig.savefig(path_output_single_cells_mutual.joinpath("pdf").joinpath(f"{cell}_z_projection_mutual.pdf"), dpi=300)



    #give each em cell that has a match above 0.1  a functional identity

    nbl_identity = nbl_all.loc[(nbl_all >= 0.1).any(axis=1),:]
    nbl_identity = nbl_identity.loc[:, (nbl_identity >= 0.1).any(axis=0)]
    nbl_identity = nbl_identity.loc[:, np.unique( nbl_identity.idxmax(axis=1))]
    nbl_functional_match = pd.DataFrame(nbl_identity.idxmax(axis=1))
    nbl_functional_match['nblast_value'] = nbl_identity.max(axis=1)
    nbl_functional_match.columns = ['pa_cell_name','nblast_value']

    projection_types_pa = ['ipsilateral','contralateral']
    neurotransmitter_pa = ['excitatory','inhibitory']

    for em_cell,pa_cell in nbl_functional_match.iterrows():
        cell_type_label = all_cells.loc[all_cells['cell_name']==pa_cell.iloc[0],"cell_type_labels"].iloc[0]
        for label in cell_type_label:
            if label in color_cell_type_dict.keys():
                nbl_functional_match.loc[em_cell,'functional_matched_identity'] = label

            if label in projection_types_pa:
                nbl_functional_match.loc[em_cell,'projection_type_pa'] = label

            if label in neurotransmitter_pa:
                nbl_functional_match.loc[em_cell,'neurotransmitter_pa'] = label

    nbl_functional_match.to_excel(path_table_output.joinpath('functional_matches_em_cells.xlsx'))

    #plot heatmap for em cell matches

    plt.figure(dpi=300)
    plt.imshow(nbl_identity)
    plt.xticks(range(len(nbl_identity.columns)), nbl_identity.columns,rotation=90)
    plt.yticks(range(len(nbl_identity.index)),nbl_identity.index,rotation=0)
    plt.gca().tick_params(axis='both', which='major', labelsize=3)
    cb = plt.colorbar()
    cb.outline.set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    for y in range(nbl_identity.shape[0]):
        max_pa_match_name = nbl_identity.iloc[y, :].idxmax()
        x = nbl_identity.columns.get_loc(max_pa_match_name)
        my_sq = np.array([[x - 0.5, y - 0.5], [x + 0.5, y - 0.5], [x + 0.5, y + 0.5], [x - 0.5, y + 0.5], [x - 0.5, y - 0.5]])
        plt.plot(my_sq[:, 0], my_sq[:, 1], color='red')
        print(x, y)
        print(max_pa_match_name)




    plt.savefig(path_heatmap_output.joinpath(r'em_cells_with_a_match_above0.1_and_partner.png'),dpi=300)
    plt.savefig(path_heatmap_output.joinpath(r'em_cells_with_a_match_above0.1_and_partner.pdf'), dpi=300)
    plt.savefig(path_heatmap_output.joinpath(r'em_cells_with_a_match_above0.1_and_partner.svg'), dpi=300)
    plt.show()





    for cell in np.concatenate([nbl_identity.columns,nbl_identity.index]):
        navis_element = all_cells.loc[all_cells['cell_name']==cell,"swc"]

        try:
            for label in all_cells.loc[all_cells['cell_name']==cell,'cell_type_labels'].iloc[0]:
                if label in color_cell_type_dict.keys():
                    color = color_cell_type_dict[label]

        except:
            color = 'k'


        fig, ax = navis.plot2d(navis_element.iloc[0], color=color, linewidth=4, method='2d', view=('x', "-y"), group_neurons=True, rasterize=True)
        fig.savefig(path_output_single_cell.joinpath("png").joinpath(f"{cell}_z_projection.png"),dpi=300)
        fig.savefig(path_output_single_cell.joinpath("svg").joinpath(f"{cell}_z_projection.svg"), dpi=300)
        fig.savefig(path_output_single_cell.joinpath("pdf").joinpath(f"{cell}_z_projection.pdf"), dpi=300)


    #cable length
    em_cl = []
    pa_cl = []
    for i,cell in all_cells.iterrows():
        if cell['imaging_modality'] == "photoactivation":
            pa_cl.append(cell['swc_smooth'].cable_length)
        else:
            em_cl.append(cell['swc_smooth'].cable_length)

    hist_em,b = np.histogram(em_cl, bins=20, range=(np.min(pa_cl+em_cl), np.max(pa_cl+em_cl)))
    hist_pa, b = np.histogram(pa_cl, bins=20, range=(np.min(pa_cl+em_cl), np.max(pa_cl+em_cl)))

    plt.bar(b[:-1], hist_em, width=1 / 140, log=False, color='orange', label='em')
    plt.bar(b[:-1], hist_pa, width=1 / 140, log=False, color='blue', label='pa')
    plt.legend()
    width = 50
    plt.bar(b[:-1], hist_pa, width=width,alpha=0.5)
    plt.bar(b[:-1] , hist_em, width=width,alpha=0.5)
    plt.title('cable length')
    plt.show()
