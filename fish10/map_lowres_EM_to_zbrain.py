from hindbrain_structure_function.mapping_helpers.ANTs_registration_helpers import ANTsRegistrationHelpers
from pathlib import Path
import os

os.environ['ANTs_use_threads'] = "11"
os.environ['ANTs_bin_path'] = "/opt/ANTs/bin"

ants_reg = ANTsRegistrationHelpers(manual_opts_dict={"debugging": False})

# # Reduce the complexity of the SyN algorithm a bit to be able to handle the larger files
# ants_reg.opts_dict["SyN"]["c"] = "[20x20x10,1e-7,10]"
# ants_reg.opts_dict["SyN"]["s"] = "4x3x2"
# ants_reg.opts_dict["SyN"]["f"] = "12x8x4"

root_dir = Path(rf'/Users/arminbahl/Nextcloud/CLEM_paper_data/em_zfish1/transforms/em_zfish1_to_zbrain_021824')

# Multi-channel registration
# ants_reg.ANTs_registration(source_path=[root_dir / "confocal_gad1b_mapped_to_lowres_EM.nrrd",
#                                         root_dir / "confocal_vglut2_mapped_to_lowres_EM.nrrd"],
#                            target_path=[root_dir / "zbrain_Gad1b-GFP_rotated_zflip.nrrd",
#                                         root_dir / "zbrain_Vglut2a-GFP_rotated_zflip.nrrd"],
#                            transformation_prefix_path=root_dir / "ANTs_dfield")

# Test the transform
# ants_reg.ANTs_applytransform(source_path=root_dir / "lowres_EM_cropped.nrrd",
#                              target_path=root_dir / "zbrain_Gad1b-GFP_rotated_zflip.nrrd",
#                              output_path=root_dir / "lowres_EM_cropped_mapped.nrrd",
#                              transformation_prefix_path=root_dir / "ANTs_dfield",
#                              use_forward_transformation=True)
#
# ants_reg.ANTs_applytransform(source_path=root_dir / "zbrain_Gad1b-GFP_rotated_zflip.nrrd",
#                              target_path=root_dir / "lowres_EM_cropped.nrrd",
#                              output_path=root_dir / "lowres_EM_cropped_mapped_inverse.nrrd",
#                              transformation_prefix_path=root_dir / "ANTs_dfield",
#                              use_forward_transformation=False)

ants_reg.ANTs_applytransform(source_path=root_dir / "confocal_gad1b_mapped_to_lowres_EM.nrrd",
                             target_path=root_dir / "zbrain_Gad1b-GFP_rotated_zflip.nrrd",
                             output_path=root_dir / "confocal_gad1b_mapped_to_zbrain.nrrd",
                             transformation_prefix_path=root_dir / "ANTs_dfield",
                             use_forward_transformation=True)
