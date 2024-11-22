from pathlib import Path
import os
from hindbrain_structure_function.mapping_helpers.ANTs_registration_helpers import ANTsRegistrationHelpers

os.environ['ANTs_use_threads'] = "11"
os.environ['ANTs_bin_path'] = "/opt/ANTs/bin"

ants_reg = ANTsRegistrationHelpers()

transformation_prefix_path = "/Users/arminbahl/Nextcloud/CLEM_paper_data/clem_zfish1/transforms/clem_zfish1_to_zbrain_022824/ANTs_dfield"

root_path = Path("/Users/arminbahl/Desktop/CLEM15_test")
cell_names = ["clem_zfish1_cell_576460752307754417"]

for cell_name in cell_names:
    input_filename = root_path / cell_name / "mapped" / f"{cell_name}_mapped.obj"
    output_filename = root_path / cell_name / "mapped" / f"{cell_name}_mapped_back_to_original.obj"

    ants_reg.ANTS_applytransform_to_obj(input_filename = input_filename,
                                       output_filename = output_filename,
                                       transformation_prefix_path = transformation_prefix_path,
                                       use_forward_transformation=False,
                                       output_scale_x=1000,
                                       output_scale_y=1000,
                                       output_scale_z=1000)
