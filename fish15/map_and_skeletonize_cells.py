from pathlib import Path
import os
from hindbrain_structure_function.mapping_helpers.ANTs_registration_helpers import ANTsRegistrationHelpers

os.environ['ANTs_use_threads'] = "11"
os.environ['ANTs_bin_path'] = "/opt/ANTs/bin"

ants_reg = ANTsRegistrationHelpers()

root_path = Path("/Users/arminbahl/Nextcloud/CLEM_paper_data")

clem_fish1_cells = ["clem_zfish1_576460752707815861"]

# clem_fish1_cells = ["clem_zfish1_8500",
#                     "clem_zfish1_576460752627812443",
#                     "clem_zfish1_576460752738878817"]

for cell_name in clem_fish1_cells:
    ants_reg.convert_synapse_file(root_path=root_path / "clem_zfish1" / "all_cells" / "test_cell",
                                  cell_name=cell_name,
                                  shift_x=0,
                                  shift_y=0,
                                  shift_z=0,
                                  scale_x=8,
                                  scale_y=8,
                                  scale_z=30,
                                  radius_set=250)  # Set a 250 nm radius

    ants_reg.map_and_skeletonize_cell(root_path=root_path / "clem_zfish1" / "all_cells" / "test_cell",
                                      cell_name=cell_name,
                                      transformation_prefix_path=root_path / "clem_zfish1" / "transforms" / "clem_zfish1_to_zbrain_022824" / "ANTs_dfield",
                                      input_limit_x=523776,  # (1024 x pixel - 1) * 512 nm x-resolution
                                      input_limit_y=327168,  # (640 y pixel - 1) * 512 nm x-resolution
                                      input_limit_z=120000,  # (251 planes - 1) * 480 nm z-resolution
                                      input_scale_x=0.001,  # The lowres stack was reduced by factor 1000, so make it ym
                                      input_scale_y=0.001,
                                      input_scale_z=0.001)
