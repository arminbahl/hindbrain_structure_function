from hindbrain_structure_function.mapping_helpers.ANTs_registration_helpers import ANTsRegistrationHelpers
from pathlib import Path
import os

os.environ['ANTs_use_threads'] = "11"
os.environ['ANTs_bin_path'] = "/opt/ANTs/bin"

ants_reg = ANTsRegistrationHelpers()

# # Reduce the complexity of the SyN algorithm a bit to be able to handle the larger files
# ants_reg.opts_dict["SyN"]["use"] = False
# ants_reg.opts_dict["SyN"]["c"] = "[20x20x10,1e-7,10]"
# ants_reg.opts_dict["SyN"]["s"] = "4x3x2"
# ants_reg.opts_dict["SyN"]["f"] = "12x8x4"

root_dir = Path(rf'/Users/arminbahl/Nextcloud/CLEM_paper_data/clem_zfish1/transforms/clem_zfish1_to_zbrain_022224')

# The initial moving transform was genated using ITK SNAP. It's critical to tell ANTS to correct orientation before it starts.
ants_reg.ANTs_registration(source_path=root_dir / "2P_volume_mapped_to_lowres_EM.nrrd",
                           target_path=root_dir / "Elavl3-H2BRFP.nrrd",
                           transformation_prefix_path=root_dir / "ANTs_dfield",
                           manual_initial_moving_transform_path=root_dir / "manual_initial_moving_transform.txt")

# Test the transform
ants_reg.ANTs_applytransform(source_path=root_dir / "2P_volume_mapped_to_lowres_EM.nrrd",
                             target_path=root_dir / "Elavl3-H2BRFP.nrrd",
                             output_path=root_dir / "2P_volume_mapped_to_zbrain.nrrd",
                             transformation_prefix_path=root_dir / "ANTs_dfield",
                             use_forward_transformation=True)

ants_reg.ANTs_applytransform(source_path=root_dir / "lowres_EM.nrrd",
                             target_path=root_dir / "Elavl3-H2BRFP.nrrd",
                             output_path=root_dir / "lowres_EM_mapped_to_zbrain.nrrd",
                             transformation_prefix_path=root_dir / "ANTs_dfield",
                             use_forward_transformation=True)

ants_reg.ANTs_applytransform(source_path=root_dir / "Elavl3-H2BRFP.nrrd",
                             target_path=root_dir / "2P_volume_mapped_to_lowres_EM.nrrd",
                             output_path=root_dir / "zbrain_mapped_to_lowres_EM.nrrd",
                             transformation_prefix_path=root_dir / "ANTs_dfield",
                             use_forward_transformation=False)
