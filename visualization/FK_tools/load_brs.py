import navis
import os
from pathlib import Path
def load_brs(base_path,which_brs='raphe',as_volume=True):
    if as_volume:
        load_type = 'volume'
    else:
        load_type = 'neuron'

    meshes = []

    for file in os.listdir(base_path.joinpath("zbrain_regions").joinpath(which_brs)):
        try:
            meshes.append(navis.read_mesh(base_path.joinpath("zbrain_regions").joinpath(which_brs).joinpath(file),
                            units='um',output=load_type))
        except:
            pass






    return meshes