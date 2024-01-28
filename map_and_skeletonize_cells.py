from pathlib import Path
import os
from analysis_helpers.analysis.utils.ANTs_registration_helpers import ANTsRegistrationHelpers

os.environ['ANTs_use_threads'] = "11"
os.environ['ANTs_bin_path'] = "/opt/ANTs/bin"

ants_reg = ANTsRegistrationHelpers()

root_path = Path("/Users/arminbahl/Desktop")

ants_reg.convert_synapse_file(root_path=root_path,
                                  cell_name='cell_002_89189')

ants_reg.map_and_skeletonize_cell(root_path=root_path,
                                  cell_name='cell_002_89189',
                                  include_synapses=True,
                                  transformation_prefix_path="/Users/arminbahl/Desktop/em_fish10_to_z_brain_011724/ANTs_dfield")

# ants_reg.map_and_skeletonize_cell(root_path,
#                                   '8500',
#                                   transformation_prefix_path="/Users/arminbahl/Desktop/em_fish15-to-z_brain_012524/ANTs_dfield",
#                                   input_limit_x=523776,  # (1024 x pixel - 1) * 512 nm x-resolution
#                                   input_limit_y=327168,  # (640 y pixel - 1) * 512 nm x-resolution
#                                   input_limit_z=120000,  # (251 planes - 1) * 480 nm z-resolution
#                                   input_scale_x=0.001,  # The lowres stack was reduced by factor 1000
#                                   input_scale_y=0.001,
#                                   input_scale_z=0.001)
