from pathlib import Path
import os
from hindbrain_structure_function.mapping_helpers.ANTs_registration_helpers import ANTsRegistrationHelpers

os.environ['ANTs_use_threads'] = "11"
os.environ['ANTs_bin_path'] = "/opt/ANTs/bin"

ants_reg = ANTsRegistrationHelpers()

transformation_prefix_path = Path("/Users/arminbahl/Nextcloud/CLEM_paper_data/em_zfish1/transforms/em_zfish1_to_zbrain_021824/ANTs_dfield")
root_path = Path("/Users/arminbahl/Desktop/EM10_test")
cell_names = ["em_fish1_126984"]

for cell_name in cell_names:

    ants_reg.convert_synapse_file(root_path=root_path,
                                  cell_name=cell_name,
                                  shift_x=4.04816196e-01 * 1000,  # Make sure the unit remains nm
                                  shift_y=5.20478002e+02 * 1000,
                                  shift_z=8.47756398e-01 * 480,
                                  scale_x=7.99082669e-03 * 1000,
                                  scale_y=-8.01760871e-03 * 1000,
                                  scale_z=6.24857731e-02 * 480,
                                  radius_set=250)  # 250 nm radius

    ants_reg.map_and_skeletonize_cell(root_path=root_path,
                                      cell_name=cell_name,
                                      transformation_prefix_path=transformation_prefix_path,
                                      input_scale_x=0.001,  # The lowres stack was reduced by factor 1000, so make it ym
                                      input_scale_y=0.001,
                                      input_scale_z=0.001,
                                      output_swap_xy=True,  # The transform maps left-ward lookfing fish, but we want upward looking
                                      output_scale_z=-1, output_shift_z=138*2  # We need to flip it in z, because the original EM stack is dorsal to ventral
                                      )
