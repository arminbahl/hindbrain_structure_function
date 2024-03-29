import navis
from pathlib import Path
def load_brs():
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
    return meshes