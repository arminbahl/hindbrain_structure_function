from pathlib import Path
from analysis_helpers.analysis.utils.ANTs_registration_helpers import ANTsRegistrationHelpers
import os

os.environ['ANTs_use_threads'] = "30"
os.environ['ANTs_bin_path'] = "/opt/ANTs/bin"

#transforms_path = Path("/Users/arminbahl/Desktop/em_fish15-to-z_brain_012524")
transforms_path = Path("/Users/arminbahl/Desktop/em_fish10_to_z_brain_011724")
#florian
input_path = Path(r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\input")
output_path = Path(r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\output")


ants_registration = ANTsRegistrationHelpers()

ants_registration.invert_bigwarp_landmarks(input_path=input_path / "bigwarp_landmarks.csv",
                                           output_path=output_path / "bigwarp_landmarks_inverse.csv")

ants_registration.convert_bigwarp_dfield_to_ANTs_dfield(dx=0.798,  # The resolution of the target
                                                        dy=0.798,
                                                        dz=2,
                                                        bigwarp_dfield_path=input_path / "bigwarp_dfield.tif",
                                                        ANTs_dfield_path=output_path / "ANTs_dfield.nii.gz")


ants_registration.convert_bigwarp_dfield_to_ANTs_dfield(dx=0.5120000,  # The resolution of the source
                                                        dy=0.5120000,
                                                        dz=0.4800000,
                                                        bigwarp_dfield_path=input_path / "bigwarp_dfield_inverse.tif",
                                                        ANTs_dfield_path=output_path / "ANTs_dfield_inverse.nii.gz")

#Test the transform
ants_registration.ANTs_applytransform(source_path=input_path / "EM_stack_lowres_cropped.nrrd",
                                      target_path=input_path / "Elavl3-H2BRFP.nrrd",
                                      output_path=output_path / "EM_stack_lowres_cropped_mapped.nrrd",
                                      transformation_prefix_path=output_path / "ANTs_dfield",
                                      use_forward_transformation=True)
#
ants_registration.ANTs_applytransform(source_path=input_path / "Elavl3-H2BRFP.nrrd",
                                      target_path=input_path / "EM_stack_lowres_cropped.nrrd",
                                      output_path=output_path / "EM_stack_lowres_cropped_mapped_inverse.nrrd",
                                      transformation_prefix_path=output_path / "ANTs_dfield",
                                      use_forward_transformation=False)


