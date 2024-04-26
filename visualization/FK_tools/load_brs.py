import navis
import os
from pathlib import Path
def load_brs(base_path,load_FK_regions=False,as_volume=True):
    if as_volume:
        load_type = 'volume'
    else:
        load_type = 'neuron'
    if load_FK_regions:
        meshes = []

        for file in os.listdir(base_path.joinpath("zbrain_regions").joinpath("FK_regions")):
            try:
                meshes.append(navis.read_mesh(base_path.joinpath("zbrain_regions").joinpath("FK_regions").joinpath(file),
                                units='um',output=load_type))
            except:
                pass


    else:
        meshes = []


        for file in os.listdir(base_path.joinpath("zbrain_regions")):
            if as_volume:
                try:
                    meshes.append(navis.read_mesh(base_path.joinpath("zbrain_regions").joinpath(file),
                                    units='um',output=load_type))
                except:
                    pass



    return meshes