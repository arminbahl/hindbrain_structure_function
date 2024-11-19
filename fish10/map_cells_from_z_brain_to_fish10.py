from pathlib import Path
import os
from hindbrain_structure_function.mapping_helpers.ANTs_registration_helpers import ANTsRegistrationHelpers

os.environ['ANTs_use_threads'] = "11"
os.environ['ANTs_bin_path'] = "/opt/ANTs/bin"

ants_reg = ANTsRegistrationHelpers()

transformation_prefix_path = "/Users/arminbahl/Nextcloud/CLEM_paper_data/em_zfish1/transforms/em_zfish1_to_zbrain_021824/ANTs_dfield"

# # Map fish 1.0 back to 1.0
# root_path = Path("/Users/arminbahl/Desktop/EM10_test")
# cell_names = ["em_fish1_126984"]

# Map some fish 1.5 cells to 1.0
# root_path = Path("/Users/arminbahl/Desktop/CLEM15_test")
# cell_names = ["clem_zfish1_cell_576460752718169177"]

# Map pa-gfp cells to 1.0
root_path = Path("/Users/arminbahl/Nextcloud/CLEM_paper_data/paGFP")
cell_names = ["20230226.1"]

for cell_name in cell_names:
    # For PA-GFP Cells
    input_filename = root_path / cell_name / f"{cell_name}_combined.obj"
    output_filename = root_path / cell_name / f"{cell_name}_combined_mapped_to_EM_original_fish10.obj"

    #input_filename = root_path / cell_name / "mapped" / f"{cell_name}_mapped.obj"
    #output_filename = root_path / cell_name / "mapped" / f"{cell_name}_mapped_to_EM_original_fish10.obj"

    ants_reg.ANTS_applytransform_to_obj(input_filename = input_filename,
                                        output_filename = output_filename,
                                        transformation_prefix_path = transformation_prefix_path,
                                        use_forward_transformation=False,
                                        input_swap_xy=True,
                                        input_scale_z=-1, input_shift_z=138*2,
                                        output_scale_x=1000,
                                        output_scale_y=1000,
                                        output_scale_z=1000)
