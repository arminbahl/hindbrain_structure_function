from analysis_helpers.analysis.utils.ANTs_registration_helpers import *
import h5py
import nrrd
import numpy as np
from analysis_helpers.analysis.personal_dirs.Florian.flo_tools.slack_bot import *
from pathlib import Path
volume = Path(r"C:\Users\ag-bahl\Downloads\2024-01-31_13-42-21_fish004_FK.nrrd")
hcr = Path(r"C:\Users\ag-bahl\Downloads\2024-04-11_10-45-16_fish002_FK.nrrd")

opts_dict = {
            'interpolation_method': 'linear',
            'ANTs_verbose': 1,
            'tempdir': None,
            'num_cores': os.environ['ANTs_use_threads'],
            'ANTs_bin_path': os.environ['ANTs_bin_path'],
            'ANTs_use-histogram-matching': 0,
            'matching_metric': 'NMI',
            "registration_order":["rigid", "affine", "SyN", "BSplineSyn"],
            "winsorize-image-intensities": "[0.005, 0.995]",
            'rigid': {"use": True,
                      "t": "Rigid[0.1]",
                      "m": "MI[$1,$2,1,32,Regular,0.25]",  # $1, $2, source and target path
                      "c": "[1000x500x250x300,1e-8,10]",
                      "s": "3x2x1x0",
                      "f": "8x4x2x1"},

            'affine': {"use": True,
                       "t": "Affine[0.1]",
                       "m": "MI[$1,$2,1,32,Regular,0.25]",  # $1, $2, source and target path
                       "c": "[200x200x200x100,1e-8,10]",
                       "s": "3x2x1x0",
                       "f": "8x4x2x1"},

            'SyN': {"use": False,
                    "t": "SyN[0.1,6,0]",  # 0.05,6,0.5 for live, and [0.1,6,0] for fixed (Marquart et al. 2017)
                    "m": "CC[$1,$2,1,2]",  # $1, $2, source and target path
                    "c": "[200x200x200x100,1e-7,10]",
                    "s": "4x3x2x1",
                    "f": "12x8x4x2"},

            'BSplineSyn': {"use": False,
                           "t": "BSplineSyn[0.1,26,0,3]",
                           "m": "CC[$1,$2,1,4]",  # $1, $2, source and target path
                           "c": "[100x70x50x20,1e-7,10]",
                           "s": "3x2x1x0",
                           "f": "6x4x2x1"}
        }
manual_initial_moving_transform_path = Path(r"C:\Users\ag-bahl\Downloads\initial_moving.txt")
def register(target, source,opts_dict,manual_initial_moving_transform_path):


    ants = ANTsRegistrationHelpers(manual_opts_dict=opts_dict)

    ants.ANTs_registration(source_path = source,
                           target_path = target,
                           manual_initial_moving_transform_path = manual_initial_moving_transform_path,
                           transformation_prefix_path=target.parent.joinpath(source.name.split('.')[0] + '_mapped2_'+ target.name.split('.')[0]))

    ants.ANTs_applytransform(source_path=source,
                             target_path=target,
                             output_path=target.parent.joinpath(source.name.split('.')[0] + '_mapped2_'+ target.name.split('.')[0]+'.nrrd'),
                             transformation_prefix_path=target.parent.joinpath(source.name.split('.')[0] + '_mapped2_'+ target.name.split('.')[0]))
    send_slack_message(MESSAGE="bspline syn finished")
register(volume,hcr,opts_dict,manual_initial_moving_transform_path)