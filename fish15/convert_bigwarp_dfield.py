from pathlib import Path
from analysis_helpers.analysis.utils.ANTs_registration_helpers import ANTsRegistrationHelpers
import os

os.environ['ANTs_use_threads'] = "11"
os.environ['ANTs_bin_path'] = "/opt/ANTs/bin"

transforms_path = Path("/Users/arminbahl/Nextcloud/CLEM_paper_data/clem_zfish1/transforms/clem_zfish1_to_zbrain_022524")

ants_registration = ANTsRegistrationHelpers()

# Maybe the new vrsion of bigwarp does create the inverse dfield already by itsself?
# ants_registration.invert_bigwarp_landmarks(input_path=transforms_path / "bigwarp_landmarks.csv",
#                                            output_path=transforms_path / "bigwarp_landmarks_inverse.csv")

# ants_registration.convert_bigwarp_dfield_to_ANTs_dfield(dx=0.798,  # The resolution of the target
#                                                         dy=0.798,
#                                                         dz=2,
#                                                         bigwarp_dfield_path=transforms_path / "bigwarp_dfield.tif",
#                                                         ANTs_dfield_path=transforms_path / "ANTs_dfield.nii.gz")


ants_registration.convert_bigwarp_dfield_to_ANTs_dfield(dx=0.5120000,  # The resolution of the source
                                                        dy=0.5120000,
                                                        dz=0.4800000,
                                                        bigwarp_dfield_path=transforms_path / "bigwarp_dfield_inverse.tif",
                                                        ANTs_dfield_path=transforms_path / "ANTs_dfield_inverse.nii.gz")
#
# ants_registration.ANTs_applytransform(source_path=transforms_path / "EM_stack_lowres.nrrd",
#                                       target_path=transforms_path / "Elavl3-H2BRFP.nrrd",
#                                       output_path=transforms_path / "EM_stack_lowres_mapped_to_zbrain.nrrd",
#                                       transformation_prefix_path=transforms_path / "ANTs_dfield",
#                                       use_forward_transformation=True)

ants_registration.ANTs_applytransform(source_path=transforms_path / "Elavl3-H2BRFP.nrrd",
                                      target_path=transforms_path / "EM_stack_lowres.nrrd",
                                      output_path=transforms_path / "Elavl3-H2BRFP_mapped_to_lowres_EM.nrrd",
                                      transformation_prefix_path=transforms_path / "ANTs_dfield",
                                      use_forward_transformation=False)


