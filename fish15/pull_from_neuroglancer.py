# This codes uploads any segment and deforms it to the zbrain 

############################################################################################################################
############################################################################################################################
import navis
import cloudvolume as cv
import numpy as np
import skeletor as sk
import trimesh as tm
import os 
import subprocess
from pathlib import Path
import plotly.offline

navis.patch_cloudvolume()
# Get Graphene token is required
vol = cv.CloudVolume('graphene://https://data.proofreading.zetta.ai/segmentation/api/v1/lichtman_zebrafish_hindbrain_001', 
                     use_https=True, progress=False)
############################################################################################################################
############################################################################################################################

# Get meshes 
root_path = Path('/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/zbrain_swc/deform_swc/')
cell_ID='8500'
soma_ID=576460752750725934
axon_ID=576460752682372578
dendrites_ID=576460752682372834

if not os.path.exists(str(root_path) + '/' + cell_ID):
    os.makedirs(str(root_path) + '/' + cell_ID)

# Get and save soma 
soma = vol.mesh.get([soma_ID], as_navis=True)
soma_path=str(root_path) + '/' + cell_ID + '/' + cell_ID + '_soma.obj'
navis.write_mesh(soma,soma_path, filetype='obj')
# Get and save axon 
axon = vol.mesh.get([axon_ID], as_navis=True)
axon_path=str(root_path) + '/' + cell_ID + '/' + cell_ID + '_axon.obj'
navis.write_mesh(axon,axon_path, filetype='obj')
# Get and save dendrites  
dendrites = vol.mesh.get([dendrites_ID], as_navis=True)
dendrites_path=str(root_path) + '/' + cell_ID + '/' + cell_ID + '_dendrites.obj'
navis.write_mesh(dendrites,dendrites_path, filetype='obj')
# Get and save the whole neuron  
neuron_parts=vol.mesh.get([soma_ID, axon_ID, dendrites_ID], as_navis=True) 
neuron_path=str(root_path) + '/' + cell_ID + '/' + cell_ID + '.obj'
neuron = navis.combine_neurons(neuron_parts)
navis.write_mesh(neuron,neuron_path, filetype='obj')
# Plot to double check 
fig = neuron.plot3d()

############################################################################################################################
############################################################################################################################

# Smooth and skeletonize objects 
segments_path = [soma_path, axon_path, dendrites_path, neuron_path]
for segment in segments_path:
    
    # Load, Fix mesh issues, save 
    mesh = tm.load_mesh(segment)
    mesh = sk.pre.fix_mesh(mesh, fix_normals=True, inplace=False)

    path_short = segment.rstrip(",.obj")  
    export_path=str(path_short) + "_simple.obj"
    mesh.export(export_path)

    #Skeletonize, save as swc 
    skel = sk.skeletonize.by_teasar(mesh, inv_dist=1500) # 1500 nm precisious skeleton
    skel = sk.post.clean_up(skel) # Remove some potential perpedicular branches that should not be there
    
    #Add a radius, save  
    sk.post.radii(skel, method='ray', validate=True) # Estimate the radius (does not really produce good results)
    swc_path=str(path_short) + "_simple.swc"
    skel.save_swc(swc_path)

    #Read and plot swc
    ku=navis.read_swc(swc_path)
    fig = navis.plot3d(ku)  

############################################################################################################################
############################################################################################################################

# Load helpers function 
from ANTs_registration_helpers import ANTsRegistrationHelpers
ants_reg = ANTsRegistrationHelpers()
transform_path= str(root_path) + 'em_zfish_to_ref_brain_Elavl3-H2BRFP_ants_dfield'

# Deform each .swc
target_dir=(str(root_path) + '/' + cell_ID)
segments_part = ['_soma', '_axon', '_dendrites', '' ]

for part in segments_part:
    segment_path=str(target_dir) + '/' + str(cell_ID) + part + '_simple.swc'
    output_path=str(target_dir) + '/' + str(cell_ID) + part + '_reg.swc'
    
    # Apply ANTs deform 
    ants_reg.ANTs_applytransform_to_swc(input_filename=segment_path,
                                        output_filename=output_path,
                                        transformation_prefix_path=transform_path,
                                        use_forward_transformation=True,
                                        input_limit_x=523776,  # (1024 x pixel - 1) * 512 nm x-resolution
                                        input_limit_y=327168,  # (640 y pixel - 1) * 512 nm x-resolution
                                        input_limit_z=120000,  # (251 planes - 1) * 480 nm z-resolution
                                        input_scale_x=0.001,
                                        input_scale_y=0.001,
                                        input_scale_z=0.001,
                                        node_size_scale=0.1,
                                        )
    
