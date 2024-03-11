import copy
import navis
import plotly
import os
from pathlib import Path
from scipy.spatial import distance
import imageio
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.cluster import KMeans
import datetime
import matplotlib
import glob
from PIL import Image, ImageDraw, ImageFont
import cv2
import os


#settings
my_lw = 0.5


cell_register = pd.read_csv(r"C:\Users\ag-bahl\Downloads\Table1.csv")
dynamic_volume = cell_register.loc[:,["Volume","Binary scored Dynamic type","Manually evaluated cell type"]]
dynamic_volume = dynamic_volume[dynamic_volume["Volume"].notna()]

dynamic_thresholds = dynamic_volume.loc[dynamic_volume['Manually evaluated cell type']=='Rebound',"Volume"].to_list()
dynamic_thresholds = [[y for y in x.split(" ")] for x in dynamic_thresholds]
dynamic_thresholds = [x for y in dynamic_thresholds for x in y]

motors = dynamic_volume.loc[dynamic_volume['Manually evaluated cell type']=='Motor',"Volume"].to_list()
motors = [[y for y in x.split(" ")] for x in motors]
motors = [x for y in motors for x in y]

integrators = dynamic_volume.loc[dynamic_volume['Manually evaluated cell type']=='Integrator',"Volume"].to_list()
integrators = [[y for y in x.split(" ")] for x in integrators]
integrators = [x for y in integrators for x in y]

integrators_group_crossing_going_caudal = ['2023-04-28_11-22-21',"2023-05-08_14-10-57",'2023-05-20_14-29-53']
integrators_group_local = ['2023-03-27_11-58-51',"2023-04-27_14-45-57",'2023-04-30_19-08-10','2023-05-13_09-13-00']
integrators_group_weird = ['2023-04-03_14-39-19','2023-04-28_16-07-27','2023-05-19_14-11-14']
integrators_group_crossing = ['2023-04-27_10-34-03','2023-05-13_16-40-52']

integrators_contra = integrators_group_crossing + integrators_group_crossing_going_caudal

rebounder_for_figure = ['2023-03-21_14-40-41']
motor_for_figure = ['2023-04-24_15-26-14']
integrator_for_figure = ['2023-04-28_11-22-21']

single_cell_to_show = ['2023-04-28_11-22-21']

selected_cells = 'all'

#selected_cells = motors
filename = 'motors'

use_johns = False


for cell_type,selected_cells in zip(['integrators_contra',"dynamic_threshold","motor",'integrators_group_local'],
                                    [integrators_contra,dynamic_thresholds,motors,integrators_group_local]):


    finished_cells_path = []
    finished_cells_swc = []
    finished_integrators_path = []
    finished_integrators_swc = []
    finished_rebounds_path = []
    finished_rebounds_swc = []
    finished_john_swc = []

    color_by_type = []
    all_neurons = False
    cluster_nblast = False
    if selected_cells == 'all':
        all_neurons = True

    for file in os.listdir(r'W:\Florian\function-neurotransmitter-morphology'):
        if Path(rf'W:\Florian\function-neurotransmitter-morphology\{file}\{file}-000_registered.nrrd').exists() and (file in selected_cells or all_neurons):
            finished_cells_path.append(Path(rf'W:\Florian\function-neurotransmitter-morphology\{file}\{file}-000_registered.swc'))
            neuron = navis.read_swc(Path(rf'W:\Florian\function-neurotransmitter-morphology\{file}\{file}-000_registered.swc'))
            neuron.soma = 1
            neuron.nodes.iloc[0, 5] = 2
            #neuron.nodes.iloc[:, 4] = 138 * 2 - neuron.nodes.iloc[:, 4]
            finished_cells_swc.append(navis.smooth_skeleton(neuron))


            temp_dynamic = dynamic_volume.loc[[file in x for x in dynamic_volume["Volume"]], "Manually evaluated cell type"].item()


            if type(file) == str:

                print(file, temp_dynamic)




            if temp_dynamic == "Integrator":
                color_by_type.append((255,0,0,1))
                finished_integrators_path.append(
                    Path(rf'W:\Florian\function-neurotransmitter-morphology\{file}\{file}-000_registered.swc'))
                finished_integrators_swc.append(navis.smooth_skeleton(neuron))

            elif temp_dynamic == "Rebound":
                    color_by_type.append((0,255,255,1))
                    finished_rebounds_path.append(
                        Path(rf'W:\Florian\function-neurotransmitter-morphology\{file}\{file}-000_registered.swc'))
                    finished_rebounds_swc.append(navis.smooth_skeleton(neuron))

            elif temp_dynamic == "Motor":
                color_by_type.append((128, 0, 128, 1))
                finished_rebounds_path.append(
                    Path(rf'W:\Florian\function-neurotransmitter-morphology\{file}\{file}-000_registered.swc'))
                finished_rebounds_swc.append(navis.smooth_skeleton(neuron))

    for file in glob.glob(r'W:\Florian\function-neurotransmitter-morphology\john_swc\*registered.swc'):
        if Path(file).exists() and use_johns:
            finished_cells_path.append(Path(file))
            neuron = navis.read_swc(Path(file))
            neuron.soma = 1
            neuron.nodes.iloc[0, 5] = 2

            temp_neuron_org = navis.smooth_skeleton(neuron)
            finished_cells_swc.append(temp_neuron_org)



            color_by_type.append((0, 0, 0, 1))

            finished_john_swc.append(navis.smooth_skeleton(neuron))
            #finished_john_swc[-1].nodes.rename(columns={"z": "x", "x": "y"}, inplace=True)




    volumes = [navis.read_mesh(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Retina.obj',
                              units='microns',output='volume'),
              navis.read_mesh(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Midbrain.obj',
                              units='microns',output='volume'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Forebrain.obj',
                  units='microns',output='volume'),
              navis.read_mesh(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Habenula.obj',
                              units='microns',output='volume'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Hindbrain.obj',
                  units='microns',output='volume'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Spinal Cord.obj',
                  units='microns',output='volume'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Gad1b Cluster 1.obj',
                  units='microns',output='volume'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Gad1b Cluster 2.obj',
                  units='microns',output='volume'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Vglut2 cluster 1.obj',
                  units='microns',output='volume'),
              navis.read_mesh(
                  Path(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Vglut2 cluster 2.obj'),
                  units='microns',output='volume'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Raphe - Superior.obj',
                  units='microns',output='volume'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Neuropil Region 6.obj',
                  units='microns',output='volume'),
              ]



    meshes = [navis.read_mesh(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Retina.obj',
                              units='microns'),
              navis.read_mesh(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Midbrain.obj',
                              units='microns'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Forebrain.obj',
                  units='microns',output='volume'),
              navis.read_mesh(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Habenula.obj',
                              units='microns'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Hindbrain.obj',
                  units='microns'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Spinal Cord.obj',
                  units='microns'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Gad1b Cluster 1.obj',
                  units='microns'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Gad1b Cluster 2.obj',
                  units='microns'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Vglut2 cluster 1.obj',
                  units='microns'),
              navis.read_mesh(
                  Path(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Vglut2 cluster 2.obj'),
                  units='microns'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Raphe - Superior.obj',
                  units='microns'),
              navis.read_mesh(
                  r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Neuropil Region 6.obj',
                  units='microns'),
              ]

    selected_meshes = [mesh.name for mesh in meshes]
    selected_meshes = ["Retina", 'Midbrain', "Forebrain", "Habenula", "Hindbrain", "Spinal Cord"]
    meshes = [mesh for mesh in meshes if mesh.name in selected_meshes]


    color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(meshes)



    fig = navis.plot3d(finished_cells_swc + meshes, backend='plotly',color=color_by_type+color_meshes,width=2500,height=1300)
    fig.update_layout(
        scene={
            'xaxis': {'autorange': 'reversed', 'range': (0, 621 * 0.798)},  # reverse !!!
            'yaxis': {'range': (0, 1406 * 0.798)},




            'zaxis': {'range': (0, 138 * 2)},
        }
    )
    plotly.offline.plot(fig, filename=rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\{cell_type}.html",auto_open=True,auto_play=False)






    #zprojection

    fig, ax = navis.plot2d(finished_cells_swc,
                            color=color_by_type,
                           alpha=1,
                           linewidth=my_lw,
                           method='2d',
                           view=('x',"-y"),
                           group_neurons = True,
                           volume_outlines=False)

    color_dict = {'Hindbrain':'#FF5448',"Midbrain":"#70C97B","Retina":'#110F4A'}
    for volume in volumes:
        if volume.name in ['Hindbrain',"Midbrain","Retina"]:
            fig, ax = navis.plot2d(volume,
                                   linewidth=my_lw,
                                   ax = ax,
                                   alpha=0.2,
                                   c = color_dict[volume.name],
                                   method='2d',
                                   view=('x',"-y"),
                                   group_neurons = True,
                                   #volume_outlines=False,
                                   rasterize = False)
    plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\for_final\z_projection_{cell_type}.pdf", dpi=300)
    plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\svg\z_projection_{cell_type}.svg", dpi=300)


    #y projection
    sublist = [volumes[4]] + volumes[6:11]

    fig, ax = navis.plot2d(finished_cells_swc,
                           color=color_by_type,
                           linewidth=my_lw,
                           method='2d',
                           view=('x',"z"),
                           group_neurons = True,
                           volume_outlines=False)

    for volume in volumes:
        if volume.name in ['Hindbrain', "Midbrain", "Retina"]:
            fig, ax = navis.plot2d(volume,
                                   c=color_dict[volume.name],
                                   linewidth=my_lw,
                                   ax=ax,
                                   alpha=0.2,
                                   method='2d',
                                   view=('x', "z"),
                                   group_neurons=True,
                                   volume_outlines=False)

    plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\for_final\y_projection_{cell_type}.pdf", dpi=300)
    plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\svg\y_projection_{cell_type}.svg", dpi=300)


    # #outlines
    # fig, ax = navis.plot2d(finished_cells_swc,
    #                        color=color_by_type,
    #                        alpha=1,
    #                        linewidth=my_lw,
    #                        method='2d',
    #                        view=('x',"z"),
    #                        group_neurons = True,
    #                        volume_outlines=False)
    #
    # fig, ax = navis.plot2d(sublist,
    #                        #color=projection_color_meshes[:6]+color_by_type,
    #                        alpha=1,
    #                        ax=ax,
    #                        linewidth=my_lw,
    #                        method='2d',
    #                        view=('x',"z"),
    #                        group_neurons = True,
    #                        volume_outlines=True)
    #
    # plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\for_final\y_projection_outlines_{cell_type}.pdf", dpi=300)
    # plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\svg\y_projection_outlines_{cell_type}.svg", dpi=300)






