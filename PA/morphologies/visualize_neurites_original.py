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

def generate_distinguishable_colors(num_colors):
    # Generate an initial random color
    colors = [np.random.rand(3)]

    while len(colors) < num_colors:
        # Generate a random color
        color = np.random.rand(3)

        # Compute the minimum distance to existing colors
        min_distance = min(distance.euclidean(color, existing_color) for existing_color in colors)

        # Only add the color if it is distinguishable from existing colors
        if min_distance > 0.3:

            colors.append(color)
    colors = [matplotlib.colors.to_hex(x) for x in colors]
    return colors


do_video = True

navis.set_pbars(hide=True)

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

rebounder_for_figure = ['2023-03-21_14-40-41']
motor_for_figure = ['2023-04-24_15-26-14']
integrator_for_figure = ['2023-04-28_11-22-21']

single_cell_to_show = ['2023-04-28_11-22-21']

selected_cells = 'all'

#selected_cells = rebounder_for_figure
filename = 'all'

use_johns = False



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




meshes = [navis.read_mesh(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Retina.obj',
                          units='microns'),
          navis.read_mesh(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Midbrain.obj',
                          units='microns'),
          navis.read_mesh(
              r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Forebrain.obj',
              units='microns'),
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


color_cells = generate_distinguishable_colors(len(finished_cells_swc))
color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(meshes)



if cluster_nblast:
    finished_cells_swc_nblast = [navis.make_dotprops(x, k=0, resample=True) for x in finished_cells_swc]
    nbl = navis.nblast(finished_cells_swc_nblast,finished_cells_swc_nblast,n_cores =1)
    nbl_array = np.array(nbl)

    plt.imshow(nbl_array, norm=matplotlib.colors.LogNorm())

    plt.colorbar()
    plt.show()

    num_clusters = 3  # Specify the number of clusters you want
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(nbl_array)
    cluster_labels = kmeans.labels_
    for no in np.unique(cluster_labels):
        fig = navis.plot3d(np.array(finished_cells_swc)[(cluster_labels==no).tolist()].tolist() + meshes, backend='plotly', color=np.array(color_by_type)[(cluster_labels==no).tolist()].tolist()  + color_meshes, width=2500,
                           height=1300)
        fig.update_layout(
            scene={
                'xaxis': {'autorange': 'reversed', 'range': (0, 621 * 0.798)},  # reverse !!!
                'yaxis': {'range': (0, 1406 * 0.798)},

                'zaxis': {'range': (0, 138 * 2)},
            }
        )
        plotly.offline.plot(fig, filename=r"C:\Users\ag-bahl\Desktop\zbrain_mesh\test.html")





fig = navis.plot3d(finished_cells_swc + meshes, backend='plotly',color=color_by_type+color_meshes,width=2500,height=1300)
fig.update_layout(
    scene={
        'xaxis': {'autorange': 'reversed', 'range': (0, 621 * 0.798)},  # reverse !!!
        'yaxis': {'range': (0, 1406 * 0.798)},




        'zaxis': {'range': (0, 138 * 2)},
    }
)
plotly.offline.plot(fig, filename=rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\{filename}.html",auto_open=True,auto_play=False)






dpi = 300
projection_zoom = 1
figsize = 500
force_new = True
#projections
projection_color_meshes = [(0.4, 0.4, 0.4, 0)] * len(meshes)
fig, ax = navis.plot2d(finished_cells_swc + meshes[:6], linewidth=1, method='3d_complex', color=color_by_type+projection_color_meshes[:6])
fig.set_dpi(dpi)


cells_in_figure = np.unique([x.name[:-15] for x in finished_cells_swc])
cell_type_in_figure = cell_register.loc[cell_register["Volume"].isin(cells_in_figure),'Manually evaluated cell type'].unique().tolist()

legend_elements= []
if "Integrator" in cell_type_in_figure:
    legend_elements.append(Patch(facecolor='#ff0000', edgecolor='#ff0000',
                             label='Integrator neuron'))
if "Rebound" in cell_type_in_figure:
    legend_elements.append(Patch(facecolor='#00ffff', edgecolor='#00ffff',
                             label='Dynamic threshold neuron'))
if "Motor" in cell_type_in_figure:
    legend_elements.append(Patch(facecolor='#800080', edgecolor='#800080',label='Motor command neuron'))
if use_johns:
    legend_elements.append(Patch(facecolor='#000000', edgecolor='#000000', label='EM integrator'))

mean_all_cells_y = np.mean([np.mean(x.nodes.loc[:,'y']) for x in finished_cells_swc])
mean_all_meshes_x = np.mean([np.mean(x.vertices[:,0]) for x in meshes])
mean_all_meshes_z = np.mean([np.mean(x.vertices[:, 0]) for x in meshes])

ax.set_ylim(mean_all_cells_y-figsize,mean_all_cells_y+figsize)
ax.set_xlim(mean_all_meshes_x+figsize, mean_all_meshes_x-figsize)
mean_zaxis = np.mean([ax.get_zlim()[1],ax.get_zlim()[0]])



#z-projection
ax.view_init(0, 0, 180, vertical_axis='y')
ax.dist = projection_zoom
plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\{filename}_z_projection.jpg", dpi=dpi)
plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\{filename}_z_projection.pdf", dpi=600)


#y-projection

ax.view_init(90, 0, 180, vertical_axis='y')
ax.dist = projection_zoom
plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\{filename}_y_projection.jpg", dpi=dpi)
plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\{filename}_y_projection.pdf", dpi=600)
#temp_image = np.array(Image.open(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\{filename}_y_projection.jpg"))

#x-projection
ax.view_init(0, 90, 180, vertical_axis='y')
ax.dist = projection_zoom
plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\{filename}_x_projection.jpg", dpi=dpi)
plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\{filename}_x_projection.pdf", dpi=600)

temp_image = np.array(Image.open(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\{filename}_x_projection.jpg"))

#xarrow
cv2.arrowedLine(temp_image,
                [int(temp_image.shape[0]*0.025),int(temp_image.shape[1]*0.975)],
                [int(temp_image.shape[0]*0.15),int(temp_image.shape[1]*0.975)],
                (0,0,0),
                thickness=5,
                line_type=cv2.LINE_AA,
                tipLength=0.05)
#y arrow
cv2.arrowedLine(temp_image,
                [int(temp_image.shape[0]*0.025),int(temp_image.shape[1]*0.975)],
                [int(temp_image.shape[0]*0.025),int(temp_image.shape[1]*0.85)],
                (0,0,0),
                thickness=5,
                line_type=cv2.LINE_AA,
                tipLength=0.05)
cv2.putText(temp_image,
            text = "Y",
            org=[int(temp_image.shape[0]*0.03275),int(temp_image.shape[0]*0.915)],
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0,0,0),
            thickness=3,
            lineType=cv2.LINE_AA)

cv2.putText(temp_image,
            text = "Z",
            org=[int(temp_image.shape[0]*0.08),int(temp_image.shape[0]*0.965)],
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0,0,0),
            thickness=3,
            lineType=cv2.LINE_AA)



imageio.imwrite(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\{filename}_x_projection_wa.jpg",temp_image)
#imageio.imwrite(rf'C:\Users\ag-bahl\Desktop\zbrain_mesh\projections\{filename}_tafelprojection.png',tafel_projections)



if do_video:
    fig, ax = navis.plot2d(finished_cells_swc + meshes, linewidth=0.5, method='3d_complex', color=color_by_type + color_meshes)
    fig.set_dpi(dpi)
    ax.legend(handles=legend_elements, loc='lower center', frameon=False)
    ax.set_ylim(mean_all_cells_y - figsize, mean_all_cells_y + figsize)
    ax.set_xlim(mean_all_meshes_x + figsize, mean_all_meshes_x - figsize)
    mean_zaxis = np.mean([ax.get_zlim()[1], ax.get_zlim()[0]])
    frames = []
    frames_filenames = []
    for i in range(0, 360, 2):
        frame_filename = rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\temp_img\frame_{i}_{filename}.jpg"
        frames_filenames.append(frame_filename)
        if force_new or not Path(frame_filename).exists():
            ax.view_init(0, i, 180, vertical_axis='y')
            ax.dist = 2.5
            plt.savefig(frame_filename, dpi=dpi)
            if i == 0:
                plt.savefig(rf"C:\Users\ag-bahl\Desktop\zbrain_mesh\temp_img\frame_{i}_{filename}.pdf", dpi=600)
            print("loading", frame_filename)
        temp_image = np.array(Image.open(frame_filename))
        frames.append(temp_image)
    imageio.mimsave(f"C:/Users/ag-bahl/Desktop/zbrain_mesh/spinning_brain/{filename}.mp4", frames, fps=30,
                    codec="libx264", output_params=["-crf", "20"])

