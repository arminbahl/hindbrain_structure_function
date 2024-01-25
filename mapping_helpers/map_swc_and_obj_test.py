from analysis_helpers.analysis.utils.ANTs_registration_helpers import ANTsRegistrationHelpers

ants_reg = ANTsRegistrationHelpers()

# ants_reg.ANTs_applytransform_to_swc(input_filename="/Users/arminbahl/Desktop/576460752671132807_simple.swc",
#                                     output_filename="/Users/arminbahl/Desktop/576460752710566176_mapped.swc",
#                                     transformation_prefix_path="/Users/arminbahl/Downloads/em_zfish_to_ref_brain_Elavl3-H2BRFP_ants_dfield",
#                                     use_forward_transformation=True,
#                                     input_limit_x=523776,  # (1024 x pixel - 1) * 512 nm x-resolution
#                                     input_limit_y=327168,  # (640 y pixel - 1) * 512 nm x-resolution
#                                     input_limit_z=120000,  # (251 planes - 1) * 480 nm z-resolution
#                                     input_scale_x=0.001,
#                                     input_scale_y=0.001,
#                                     input_scale_z=0.001,
#                                     node_size_scale=0.1,
#                                     )

ants_reg.ANTS_applytransform_to_obj(input_filename="/Users/arminbahl/Downloads/576460752710566176_full2.obj",
                                    output_filename="/Users/arminbahl/Downloads/576460752710566176_full2_mapped.obj",
                                    transformation_prefix_path="/Users/arminbahl/Downloads/em_zfish_to_ref_brain_Elavl3-H2BRFP_ants_dfield",
                                    use_forward_transformation=True,
                                    input_limit_x=523776,  # (1024 x pixel - 1) * 512 nm x-resolution
                                    input_limit_y=327168,  # (640 y pixel - 1) * 512 nm x-resolution
                                    input_limit_z=120000,  # (251 planes - 1) * 480 nm z-resolution
                                    input_scale_x=None,#0.001,
                                    input_scale_y=None, #0.001,
                                    input_scale_z=None)##0.001)








