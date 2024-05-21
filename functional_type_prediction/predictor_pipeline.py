import os

import matplotlib.pyplot as plt
import scipy
from hindbrain_structure_function.functional_type_prediction.FK_tools.load_cells_predictor_pipeline import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.nblast import *
from hindbrain_structure_function.functional_type_prediction.FK_tools.make_dendrogramms import *
import winsound

from datetime import datetime
import plotly
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":



    name_time = datetime.now()
    # Set the base path for data by reading from a configuration file; ensures correct data location is used.
    path_to_data = get_base_path()  # Ensure this path is set in path_configuration.txt

    all_cells = load_cells_predictor_pipeline(path_to_data=path_to_data,modalities=['pa',"clem"]) #

    # Define a dictionary mapping cell types to specific RGBA color codes for consistent visual representation.
    color_cell_type_dict = {
        "integrator_ipsi": (254, 179, 38, 0.7),
        "integrator_contra": (232, 77, 138, 0.7),
        "dynamic threshold": (100, 197, 235, 0.7),
        "motor command": (127, 88, 175, 0.7),
    }

    all_cells = all_cells[~all_cells['swc'].isna()]

    # fig = navis.plot3d([x for x in all_cells.iloc[:, -1]], backend='plotly',
    #                    width=3840, height=2160, hover_name=True, hover_id=False)
    # fig.update_layout(
    #     scene={
    #         'xaxis': {'autorange': 'reversed', 'zeroline': False, 'visible': False},  # reverse !!!
    #         'yaxis': {'autorange': True, 'zeroline': False, 'visible': False},
    #
    #         'zaxis': {'autorange': True, 'zeroline': False, 'visible': False},
    #         'aspectmode': "data",
    #         'aspectratio': {"x": 1, "y": 1, "z": 1}
    #     }
    # )
    # fig.update_layout(showlegend=False)
    # plotly.offline.plot(fig, filename="test.html", auto_open=True, auto_play=False)


    #split up cell_type_labels in own columns
    cell_type_categories = {'morphology':['ipsilateral','contralateral'],
                            'neurotransmitter':['inhibitory','excitatory'],
                            'function':['integrator','dynamic_threshold','dynamic threshold','motor_command','motor command']}


    for i,cell in all_cells.iterrows():
        if type(cell.cell_type_labels) == list:
            for label in cell.cell_type_labels:
                if label in cell_type_categories['morphology']:
                    all_cells.loc[i,'morphology'] = label
                elif label in cell_type_categories['function']:
                    all_cells.loc[i,'function'] = label
                elif label in cell_type_categories['neurotransmitter']:
                    all_cells.loc[i,'neurotransmitter'] = label


    #compare within and between groups

    # within_ii,between_ii  = compute_nblast_within_and_between(all_cells,['ipsilateral','integrator'])
    # within_eii, between_eii = compute_nblast_within_and_between(all_cells, ['ipsilateral', 'integrator','excitatory'])
    #
    # within_ci,between_ci  = compute_nblast_within_and_between(all_cells,['contralateral','integrator'])
    # within_ici, between_ici = compute_nblast_within_and_between(all_cells, ['contralateral', 'integrator','inhibitory'])
    #
    # within_dt,between_dt  = compute_nblast_within_and_between(all_cells,['dynamic_threshold'])
    # within_mc, between_mc = compute_nblast_within_and_between(all_cells, ['motor_command'])
    #
    # all_whithin_between_values = [within_ii,between_ii,within_eii, between_eii,within_ci,between_ci,within_ici, between_ici,within_dt,between_dt,within_mc, between_mc]
    # all_whithin_between_values_str = ['within_ii', 'between_ii', 'within_eii', 'between_eii', 'within_ci', 'between_ci', 'within_ici', 'between_ici', 'within_dt', 'between_dt', 'within_mc', 'between_mc']
    #
    #
    # #plot barplots
    # plt.figure()
    # y_loc = np.max(np.concatenate(all_whithin_between_values)) * 1.1
    # for i,values in enumerate(all_whithin_between_values):
    #     plt.boxplot(values,positions=[i])
    #     if i %2 == 0:
    #         plt.plot([i, i + 1], [y_loc, y_loc], lw=3, c='k')
    #
    #
    #         plt.show()
    #         last = values
    #     elif i %2 == 1:
    #         t_stat, p_value = scipy.stats.ttest_ind(values,last)
    #         stat, p_value = scipy.stats.mannwhitneyu(values, last)
    #         sig = 'n.s.'
    #         if p_value < 0.05:
    #             sig = '*'
    #         if p_value < 0.01:
    #             sig = '**'
    #         if p_value < 0.001:
    #             sig = '***'
    #
    #         plt.text(i-0.5, y_loc, sig, ha='center', va='bottom',fontsize=15)
    #
    # plt.xticks(range(len(all_whithin_between_values_str)),labels=all_whithin_between_values_str,rotation=25)
    # plt.show()

    # make_dendrogramms(path_to_data,all_cells)
    if len(np.unique(all_cells.loc[:, "swc"].apply(lambda x: x.sampling_resolution)) )!= 1:

        all_cells.loc[:, "swc"] = all_cells.loc[:, "swc"].apply(lambda x: x.resample(0.1,inplace=False))

    #add cable length
    all_cells.loc[:, 'cable_length'] = all_cells.loc[:,"swc"].apply(lambda x: x.cable_length)
    #add bbox volume
    all_cells.loc[:, 'bbox_volume'] = all_cells.loc[:, "swc"].apply(lambda x: (x.extents[0]) * (x.extents[1]) * (x.extents[2]))
    # add x_extenct
    all_cells.loc[:, 'x_extent'] = all_cells.loc[:, "swc"].apply(lambda x: x.extents[0])
    # add y_extenct
    all_cells.loc[:, 'y_extent'] = all_cells.loc[:, "swc"].apply(lambda x: x.extents[1])
    # add z_extenct
    all_cells.loc[:, 'z_extent'] = all_cells.loc[:, "swc"].apply(lambda x: x.extents[2])

    #add n_leafs
    all_cells.loc[:,'n_leafs'] = all_cells.loc[:, "swc"].apply(lambda x: x.n_leafs)
    # add n_branches
    all_cells.loc[:, 'n_branches'] = all_cells.loc[:, "swc"].apply(lambda x: x.n_branches)
    #add n_ends
    all_cells.loc[:,"n_ends"] = all_cells.loc[:, "swc"].apply(lambda x: x.n_ends)
    #add n_edges
    all_cells.loc[:,"n_edges"] = all_cells.loc[:, "swc"].apply(lambda x: x.n_edges)
    #main brainchpoint
    all_cells.loc[:, "main_branchpoint"] = all_cells.loc[:, "swc"].apply(lambda x: navis.find_main_branchpoint(x))

    # number of persitence points n_persistence_points
    all_cells.loc[:, "n_persistence_points"] =all_cells.loc[:, "swc"].apply(lambda x: len(navis.persistence_points(x)))
    #add strahler index
    _ =all_cells.loc[:, "swc"].apply(lambda x: navis.strahler_index(x))
    #add max strahler index
    all_cells.loc[:, "max_strahler_index"] = all_cells.loc[:, "swc"].apply(lambda x: x.nodes.strahler_index.max())

    #add sholl distance most bracnhes
    all_cells.loc[:,"sholl_distance_max_branches"] = all_cells.loc[:, "swc"].apply(lambda x: navis.sholl_analysis(x, radii=np.arange(10, 200, 5), center='root').branch_points.idxmax())

    # add sholl distance most bracnhes
    all_cells.loc[:, "sholl_distance_max_branches"] = all_cells.loc[:, "swc"].apply(lambda x: navis.sholl_analysis(x, radii=np.arange(10, 200, 10), center='root').branch_points.idxmax())
    all_cells.loc[:, "sholl_distance_max_branches_cable_length"] = all_cells.loc[:, ['sholl_distance_max_branches',"swc"]].apply(lambda x: navis.sholl_analysis(x['swc'],radii= np.arange(10,200,10),center='root', geodesic=False).cable_length[x['sholl_distance_max_branches']],axis=1)
    # add sholl distance most bracnhes
    all_cells.loc[:, "sholl_distance_max_branches_geosidic"] = all_cells.loc[:, "swc"].apply(lambda x: navis.sholl_analysis(x, radii=np.arange(10, 200, 10), center='root', geodesic=True).branch_points.idxmax())
    all_cells.loc[:, "sholl_distance_max_branches_geosidic_cable_length"] = all_cells.loc[:, ['sholl_distance_max_branches_geosidic',"swc"]].apply(lambda x: navis.sholl_analysis(x['swc'],radii= np.arange(10,200,10),center='root', geodesic=False).cable_length[x['sholl_distance_max_branches_geosidic']],axis=1)




    winsound.Beep(440, 500)





