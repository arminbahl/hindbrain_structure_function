
# Creates html plots and rotating videos for mapped neurons 
# Install environnement using conda env create --file make_videos.yaml
# Version: 0.2 05/03/2024 jbw

import navis
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import numpy as np
import plotly
import plotly.offline

## Generate the html visulizations
root_meshes='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/meshes_zbrain/'
meshes=[navis.read_mesh((root_meshes + 'Retina.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Midbrain.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Forebrain.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Habenula.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Forebrain.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Hindbrain.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'non-mece_Hindbrain - Vglut2 cluster 1.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'non-mece_Hindbrain - Vglut2 cluster 2.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'non-mece_Hindbrain - Gad1b Cluster 1.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'non-mece_Hindbrain - Gad1b Cluster 1.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'Raphe - Superior.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'Spinal Cord.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'non-mece_Hindbrain - Neuropil Region 6.obj'), units='microns'),
]
color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(meshes)
color_meshes[0] = (0.4, 0.4, 0.4, 0.05)
color_meshes[-1] = (0.4, 0.0, 0.0, 0.2)
color_meshes[-2] = (0.0, 0.0, 0.4, 0.2)
color_meshes[-3] = (0.0, 0.4, 0.4, 0.2)
color_meshes[-4] = (0.6, 0.2, 0.4, 0.2)
#color_meshes[-5] = (0.4, 0.2, 0.6, 0.2)
color_meshes[-6] = (0.4, 0.4, 0.5, 0.2)

root_cells='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/all_cells/'
swc_to_plot = [navis.read_swc(root_cells + 'clem_zfish1_576460752680588674/mapped/clem_zfish1_576460752680588674_mapped.swc'), 
                navis.read_swc(root_cells + 'clem_zfish1_576460752697529306/mapped/clem_zfish1_576460752697529306_mapped.swc'),
                navis.read_swc(root_cells + 'clem_zfish1_576460752707815861/mapped/clem_zfish1_576460752707815861_mapped.swc'),
]
color_cells = [(0.941, 0.122, 0.122),(0.929, 0.243, 0.906), (0.18, 0.949, 0.122)]

#Plot the swcs and main region outlines
fig = navis.plot3d(meshes+swc_to_plot, backend='plotly',alpha=0.05, linewidth=3, color=color_meshes + color_cells)
# Plot as separate html in a new window for save/export 
fig = navis.plot3d(meshes+swc_to_plot, backend='plotly',alpha=0.05, linewidth=2, color=color_meshes + color_cells, inline=False, width=1800, height=800)
_ = plotly.offline.plot(fig, filename="/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/visualization/example_plot.html") 

###########################
###########################

## Make objs. orthogonal views with hindbrain outlines
root_cells='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/all_cells/'
root_meshes='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/meshes_zbrain/'

objs_to_plot = [navis.read_mesh(root_cells + 'clem_zfish1_576460752680588674/mapped/clem_zfish1_576460752680588674_mapped.obj'), 
                navis.read_mesh(root_cells + 'clem_zfish1_576460752697529306/mapped/clem_zfish1_576460752697529306_mapped.obj'),
                navis.read_mesh(root_cells + 'clem_zfish1_576460752707815861/mapped/clem_zfish1_576460752707815861_mapped.obj'),
                navis.read_mesh(root_meshes + 'Hindbrain.obj', units='microns'),
                navis.read_mesh(root_meshes + 'Cerebellar Corpus.obj', units='microns'),
]
color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(objs_to_plot)
color_meshes[0] = (0.941, 0.122, 0.122, 1)
color_meshes[1] = (0.929, 0.243, 0.906, 1)
color_meshes[2] = (0.18, 0.949, 0.122, 1)
color_meshes[3] = (0.4, 0.4, 0.4, 0.03)
color_meshes[4] = (0.4, 0.4, 0.4, 0.04)
#Coronal view
fig, ax1 = navis.plot2d(objs_to_plot, method='2d', color=color_meshes, view=('x', 'z'))
fig.savefig('coronal.png', dpi=1200)
#Dorsal view
fig, ax1 = navis.plot2d(objs_to_plot, method='2d', color=color_meshes, view=('x', '-y'))
fig.savefig('dorsal.png', dpi=1200)
#Sagital view
fig, ax1 = navis.plot2d(objs_to_plot, method='2d', color=color_meshes, view=('-y', 'z'))
fig.savefig('saggital.png', dpi=1200)

## Make lateral view with hindbrain outlines with obj with axons/dendrites differentiated
# For each cell pull each subparts and plot with corresponding colors 
root_cells='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/all_cells/'
root_meshes='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/meshes_zbrain/'

objs_to_plot = [navis.read_mesh(root_cells + 'clem_zfish1_576460752680588674/mapped/clem_zfish1_576460752680588674_soma_mapped.obj'), 
                navis.read_mesh(root_cells + 'clem_zfish1_576460752697529306/mapped/clem_zfish1_576460752697529306_soma_mapped.obj'),
                navis.read_mesh(root_cells + 'clem_zfish1_576460752707815861/mapped/clem_zfish1_576460752707815861_soma_mapped.obj'),

                navis.read_mesh(root_cells + 'clem_zfish1_576460752680588674/mapped/clem_zfish1_576460752680588674_dendrite_mapped.obj'), 
                navis.read_mesh(root_cells + 'clem_zfish1_576460752697529306/mapped/clem_zfish1_576460752697529306_dendrite_mapped.obj'),
                navis.read_mesh(root_cells + 'clem_zfish1_576460752707815861/mapped/clem_zfish1_576460752707815861_dendrite_mapped.obj'),

                navis.read_mesh(root_cells + 'clem_zfish1_576460752680588674/mapped/clem_zfish1_576460752680588674_axon_mapped.obj'), 
                navis.read_mesh(root_cells + 'clem_zfish1_576460752697529306/mapped/clem_zfish1_576460752697529306_axon_mapped.obj'),
                navis.read_mesh(root_cells + 'clem_zfish1_576460752707815861/mapped/clem_zfish1_576460752707815861_axon_mapped.obj'),

                navis.read_mesh(root_meshes + 'Hindbrain.obj', units='microns'),
                navis.read_mesh(root_meshes + 'Cerebellar Corpus.obj', units='microns'),
]

color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(objs_to_plot)
color_meshes[0] = (0, 0, 0, 1)
color_meshes[1] = (0, 0, 0, 1)
color_meshes[2] = (0, 0, 0, 1)
color_meshes[3] = (0.941, 0.122, 0.122, 1)
color_meshes[4] = (0.929, 0.243, 0.906, 1)
color_meshes[5] = (0.18, 0.949, 0.122, 1)
color_meshes[6] = (0, 0, 0, 1)
color_meshes[7] = (0, 0, 0, 1)
color_meshes[8] = (0, 0, 0, 1)
color_meshes[9] = (0.4, 0.4, 0.4, 0.03)
color_meshes[10] = (0.4, 0.4, 0.4, 0.04)

#Coronal view
fig, ax1 = navis.plot2d(objs_to_plot, method='2d', color=color_meshes, view=('x', 'z'))
fig.savefig('neuron_parts_coronal.png', dpi=1200)
#Dorsal view
fig, ax1 = navis.plot2d(objs_to_plot, method='2d', color=color_meshes, view=('x', '-y'))
fig.savefig('neuron_parts_dorsal.png', dpi=1200)
#Sagital view
fig, ax1 = navis.plot2d(objs_to_plot, method='2d', color=color_meshes, view=('-y', 'z'))
fig.savefig('neuron_parts_saggital.png', dpi=1200)

## Make lateral view with hindbrain outlines with swc with axons/dendrites differentiated
# For each cell pull each subparts and plot with corresponding colors 
root_cells='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/all_cells/'
root_meshes='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/meshes_zbrain/'

meshes=[navis.read_mesh(root_meshes + 'Hindbrain.obj', units='microns'),
        navis.read_mesh(root_meshes + 'Cerebellar Corpus.obj', units='microns')]
swc_to_plot = [navis.read_swc(root_cells + 'clem_zfish1_576460752680588674/mapped/clem_zfish1_576460752680588674_mapped.swc'), 
               navis.read_swc(root_cells + 'clem_zfish1_576460752697529306/mapped/clem_zfish1_576460752697529306_mapped.swc'),
               navis.read_swc(root_cells + 'clem_zfish1_576460752707815861/mapped/clem_zfish1_576460752707815861_mapped.swc'),
]
color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(meshes)
color_meshes[0] = (0.4, 0.4, 0.4, 0.03)
color_meshes[1] = (0.4, 0.4, 0.4, 0.04)
##
color_cells = [(0.4, 0.4, 0.4)] * len(swc_to_plot)
color_cells[0] = (0.941, 0.122, 0.122)
color_cells[1] = (0.929, 0.243, 0.906)
color_cells[2] = (0.18, 0.949, 0.122)

#Coronal view
fig, ax1 = navis.plot2d(meshes+swc_to_plot,  method='2d', linewidth=0.25, color=color_meshes + color_cells, view=('x', 'z'))
fig.savefig('neuron_parts_coronal_swc.png', dpi=1200)
#Dorsal view
fig, ax1 = navis.plot2d(meshes+swc_to_plot, method='2d', linewidth=0.25, color=color_meshes + color_cells, view=('x', '-y'))
fig.savefig('neuron_parts_dorsal_swc.png', dpi=1200)
#Sagital view
fig, ax1 = navis.plot2d(meshes+swc_to_plot, method='2d', linewidth=0.25, color=color_meshes + color_cells, view=('-y', 'z'))
fig.savefig('neuron_parts_saggital_swc.png', dpi=1200)

####################### 
####################### 

## Make video selected set of cells

# Meshes info
root_meshes='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/meshes_zbrain/'
meshes=[navis.read_mesh((root_meshes + 'Retina.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Midbrain.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Forebrain.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Habenula.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Forebrain.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Hindbrain.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'non-mece_Hindbrain - Vglut2 cluster 1.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'non-mece_Hindbrain - Vglut2 cluster 2.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'non-mece_Hindbrain - Gad1b Cluster 1.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'non-mece_Hindbrain - Gad1b Cluster 1.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'Raphe - Superior.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'Spinal Cord.obj'), units='microns'),
        navis.read_mesh((root_meshes + 'non-mece_Hindbrain - Neuropil Region 6.obj'), units='microns'),
]
color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(meshes)
color_meshes[0] = (0.4, 0.4, 0.4, 0.05)
color_meshes[-1] = (0.4, 0.0, 0.0, 0.2)
color_meshes[-2] = (0.4, 0.4, 0.4, 0.1)
color_meshes[-3] = (0.0, 0.4, 0.4, 0.2)
color_meshes[-4] = (0.6, 0.2, 0.4, 0.2)
color_meshes[-6] = (0.4, 0.4, 0.5, 0.2)

# Cells info
swc_to_plot = [navis.read_swc(root_cells + 'clem_zfish1_576460752680588674/mapped/clem_zfish1_576460752680588674_mapped.swc'), 
               navis.read_swc(root_cells + 'clem_zfish1_576460752697529306/mapped/clem_zfish1_576460752697529306_mapped.swc'),
               navis.read_swc(root_cells + 'clem_zfish1_576460752707815861/mapped/clem_zfish1_576460752707815861_mapped.swc'),
]
color_cells = [(0.4, 0.4, 0.4)] * len(swc_to_plot)
color_cells[0] = (0.941, 0.122, 0.122)
color_cells[1] = (0.929, 0.243, 0.906)
color_cells[2] = (0.18, 0.949, 0.122)

# Initialize video type 
writer = imageio.get_writer("/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/visualization/example_video.mp4",
                            fps=24,
                            codec="libx264",
                            output_params=["-crf", "20"],
                            ffmpeg_log_level="error")

#Cells+Mesh plot
fig, ax1 = navis.plot2d(meshes + swc_to_plot, linewidth=0.25, method='3d_complex', color=color_meshes + color_cells)
# Change the background color, as there are no transparent videos
ax1.set_facecolor("white")
ax1.patch.set_alpha(1)

# Render 3D rotation
for alpha in range(0, 360, 5):
    ax1.view_init(0, alpha, 180, vertical_axis='y')
    fig.savefig('/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/visualization/temp.png', dpi=600, transparent=False)
    # Open the image file
    image = np.array(Image.open('/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/visualization/temp.png'))
    writer.append_data(image) 
writer.close()

####################### 
####################### 
## Orig
writer = imageio.get_writer("/Users/arminbahl/Desktop/test_new.mp4",
                            fps=24,
                            codec="libx264",
                            output_params=["-crf", "20"],
                            ffmpeg_log_level="error")

# USe obj files, exported from https://rordenlab.github.io/nii2meshWeb/
# After conversion, make sure, the x and y mesh coordinate are positive!!
meshes = [navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_mece/Retina.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_mece/Midbrain.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_mece/Forebrain.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_mece/Habenula.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_mece/Forebrain.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_mece/Hindbrain.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_mece/Spinal Cord.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_z_brain_1_0/Rhombencephalon - Gad1b Cluster 1.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_z_brain_1_0/Rhombencephalon - Gad1b Cluster 2.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_z_brain_1_0/Rhombencephalon - Vglut2 cluster 1.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_z_brain_1_0/Rhombencephalon - Vglut2 cluster 2.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_z_brain_1_0/Rhombencephalon - Raphe - Superior.obj', units='microns'),
          navis.read_mesh('/Volumes/ag-bahl_general/Zebrafish atlases/z_brain_atlas/region_masks_z_brain_1_0/Rhombencephalon - Neuropil Region 6.obj', units='microns'),
          ]

cells = [navis.read_mesh("/Users/arminbahl/Desktop/cell_002_89189/cell_002_89189_mapped.obj")]

         #navis.read_swc("/Users/arminbahl/Downloads/576460752710566176.swc")
         #]

color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(meshes)
color_meshes[0] = (0.4, 0.4, 0.4, 0.05)
color_meshes[-1] = (0.4, 0.0, 0.0, 0.2)
color_meshes[-2] = (0.0, 0.0, 0.4, 0.2)
color_meshes[-3] = (0.0, 0.4, 0.4, 0.2)
color_meshes[-4] = (0.6, 0.2, 0.4, 0.2)
color_meshes[-5] = (0.4, 0.2, 0.6, 0.2)
color_meshes[-6] = (0.4, 0.4, 0.5, 0.2)
color_cells = ["red", 'blue', 'green']

# Soma need to be labled with =1 in the beginnign of swc and need to have a radius,
# Follow specifications of swc standard:
# http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
# cells[0].soma = 1
# cells[0].soma_radius = 1
# cells[1].soma = 1
# cells[1].soma_radius = 1

#print(cells[0].soma_pos)
#print(cells[1].soma_pos)

# For interactive plotting in html
nl = navis.NeuronList(meshes + cells)

viewer = nl.plot3d(backend='plotly', inline=True, color=color_meshes + color_cells)
viewer.update_layout(
    scene={
        'xaxis': {'autorange': 'reversed', 'range':(0, 621*0.798) },  # reverse !!!
        'yaxis': {'range': (0, 1406*0.798)},
        'zaxis': {'range': (0, 138*2)},
    }
)

plotly.offline.plot(viewer, filename='/Users/arminbahl/Desktop/test.html')
jjjj
# Make video
fig, ax1 = navis.plot2d(meshes + cells, linewidth=0.25, method='3d_complex', color=color_meshes + color_cells)
ax1.set_xlim([0, 621 * 0.798])
ax1.set_ylim([0, 1406 * 0.798])
ax1.set_zlim([0, 138 * 2])

# Change the background color, as there are no transparent videos
ax1.set_facecolor("white")
ax1.patch.set_alpha(1)

# Render 3D rotation
for alpha in range(0, 360, 5):
    ax1.view_init(0, alpha, 180, vertical_axis='y')
    ax1.set_box_aspect([138 * 2, 1406 * 0.798, 621 * 0.798], zoom=1)

    plt.draw()
    plt.savefig('/Users/arminbahl/Desktop/fgva.png', dpi=300, transparent=False)

    # Open the image file
    image = np.array(Image.open('/Users/arminbahl/Desktop/fgva.png'))
    writer.append_data(image)

writer.close()
plt.close()

####

# Make a video for external display
import plotly.offline
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import plotly

# Export properties
vid_path = str(target_dir) + "/" + str(cell_ID) + ".mp4"
writer = imageio.get_writer(
    vid_path,
    fps=24,
    codec="libx264",
    output_params=["-crf", "20"],
    ffmpeg_log_level="error",
)

# Plot deformed neuron parts over the zbrain regioins
# Load brain outlines
meshes = [
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/Retina.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/Midbrain.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/Forebrain.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/Habenula.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/Forebrain.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/Hindbrain.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/Spinal Cord.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/non-mece_Hindbrain - Gad1b Cluster 1.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/non-mece_Hindbrain - Gad1b Cluster 2.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/non-mece_Hindbrain - Vglut2 cluster 1.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/non-mece_Hindbrain - Vglut2 cluster 2.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/Raphe - Superior.obj",
        units="microns",
    ),
    navis.read_mesh(
        "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/meshes_zbrain/non-mece_Hindbrain - Neuropil Region 6.obj",
        units="microns",
    ),
]

# Load the deformed neurons parts
soma_path = str(target_dir) + "/" + str(cell_ID) + "_soma_reg.swc"
axon_path = str(target_dir) + "/" + str(cell_ID) + "_axon_reg.swc"
dendrites_path = str(target_dir) + "/" + str(cell_ID) + "_dendrites_reg.swc"
cells = [
    navis.read_swc(soma_path),
    navis.read_swc(axon_path),
    navis.read_swc(dendrites_path),
]

# Plot options
color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(meshes)
color_meshes[0] = (0.4, 0.4, 0.4, 0.05)
color_meshes[-1] = (0.4, 0.0, 0.0, 0.2)
color_meshes[-2] = (0.0, 0.0, 0.4, 0.2)
color_meshes[-3] = (0.0, 0.4, 0.4, 0.2)
color_meshes[-4] = (0.6, 0.2, 0.4, 0.2)
color_meshes[-5] = (0.4, 0.2, 0.6, 0.2)
color_meshes[-6] = (0.4, 0.4, 0.5, 0.2)
color_cells = ["red", "blue", "green"]

# Soma need to be labled with =1 in the beginnign of swc and need to have a radius,
# Follow specifications of swc standard:
# http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
cells[0].soma = 1
cells[0].soma_radius = 1
cells[1].axon = 1
cells[1].axon_radius = 1

# For interactive plotting in html
nl = navis.NeuronList(meshes + cells)
viewer = nl.plot3d(backend="plotly", inline=True, color=color_meshes + color_cells)

html_path = str(target_dir) + "/" + str(cell_ID) + ".html"
plotly.offline.plot(viewer, html_path)

import plotly.offline

# Plot as separate html in a new window for save/export
fig = navis.plot3d(
    cells + meshes,
    backend="plotly",
    alpha=0.05,
    linewidth=3,
    inline=False,
    width=1800,
    height=800,
)
_ = plotly.offline.plot(fig)

# Save
plotly.offline.plot(
    viewer, filename=str(root_path) + "/" + cell_ID + "/" + cell_ID + ".html"
)

############################################################################################################################
############################################################################################################################

# Here
# Add the functional recordings