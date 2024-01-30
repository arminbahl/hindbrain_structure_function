from pathlib import Path
import os
from hindbrain_structure_function.mapping_helpers.ANTs_registration_helpers import ANTsRegistrationHelpers
import trimesh as tm

mesh = tm.load("/Users/arminbahl/Desktop/cell_007_124806/cell_007_124806_soma.obj")

os.environ['ANTs_use_threads'] = "11"
os.environ['ANTs_bin_path'] = "/opt/ANTs/bin"

ants_reg = ANTsRegistrationHelpers()

root_path = Path("/Users/arminbahl/Desktop")

ants_reg.convert_synapse_file(root_path=root_path,
                              cell_name="cell_002_89189",
                              shift_x=4.04816196e-01 * 1000,  # Make sure the unit remains nm
                              shift_y=5.20478002e+02 * 1000,
                              shift_z=8.47756398e-01 * 480,
                              scale_x=7.99082669e-03 * 1000,
                              scale_y=-8.01760871e-03 * 1000,
                              scale_z=6.24857731e-02 * 480,
                              radius_set=250)  # 250 nm radius

ants_reg.map_and_skeletonize_cell(root_path=root_path,
                                  cell_name="cell_002_89189",
                                  transformation_prefix_path="/Users/arminbahl/Desktop/em_fish10_to_z_brain_011724/ANTs_dfield",
                                  input_scale_x=0.001,  # Convert all units from um to nm
                                  input_scale_y=0.001,
                                  input_scale_z=0.001)

ants_reg.map_and_skeletonize_cell(root_path=root_path,
                                  cell_name="cell_007_124806",
                                  transformation_prefix_path="/Users/arminbahl/Desktop/em_fish10_to_z_brain_011724/ANTs_dfield",
                                  input_scale_x=0.001,  # Convert all units from um to nm
                                  input_scale_y=0.001,
                                  input_scale_z=0.001)

ants_reg.map_and_skeletonize_cell(root_path,
                                  '8500',
                                  transformation_prefix_path="/Users/arminbahl/Desktop/em_fish15-to-z_brain_012524/ANTs_dfield",
                                  input_limit_x=523776,  # (1024 x pixel - 1) * 512 nm x-resolution
                                  input_limit_y=327168,  # (640 y pixel - 1) * 512 nm x-resolution
                                  input_limit_z=120000,  # (251 planes - 1) * 480 nm z-resolution
                                  input_scale_x=0.001,  # The lowres stack was reduced by factor 1000
                                  input_scale_y=0.001,
                                  input_scale_z=0.001)
