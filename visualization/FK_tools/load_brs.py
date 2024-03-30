import navis
import os
from pathlib import Path
def load_brs(base_path,load_FK_regions=False):
    if load_FK_regions:
        meshes = []

        for file in os.listdir(base_path.joinpath("zbrain_regions").joinpath("FK_regions")):
            try:
                meshes.append(navis.read_mesh(base_path.joinpath("zbrain_regions").joinpath("FK_regions").joinpath(file),
                                units='microns'))
            except:
                pass

    else:
        meshes = []
        try:
            for file in os.listdir(base_path.joinpath("zbrain_regions")):
                meshes.append(navis.read_mesh(base_path.joinpath("zbrain_regions").joinpath(file),
                                units='microns'))
        except:
            pass

    return meshes