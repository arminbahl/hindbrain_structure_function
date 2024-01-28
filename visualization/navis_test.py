import navis
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import numpy as np
import plotly

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
