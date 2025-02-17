import h5py
import numpy as np
from tifffile import imsave
import pandas as pd
import warnings
import csv

container_functional_hdf5 = h5py.File("/Users/arminbahl/Desktop/unit_centroids_ants_registered.hdf5", "r")

all_masks_indexed_hdf5_mapzebrain = h5py.File(r"/Users/arminbahl/Desktop/mapzebrain_in_zbrain_coordinates/mapzebrain.hdf5", "r")
all_masks_indexed_hdf5_z_brain10 = h5py.File(r"/Users/arminbahl/Desktop/mapzebrain_in_zbrain_coordinates/z_brain10.hdf5", "r")

print("Make a huge array of 1s and 0s, for each brain region")

masks_mapzebrain = np.zeros((len(all_masks_indexed_hdf5_mapzebrain.keys()), 138, 1406, 621), dtype=bool)
masks_zbrain10 = np.zeros((len(all_masks_indexed_hdf5_z_brain10.keys()), 138, 1406, 621), dtype=bool)

for i, key in enumerate(all_masks_indexed_hdf5_mapzebrain.keys()):
    masks_mapzebrain[i][tuple(all_masks_indexed_hdf5_mapzebrain[key]["ind_mask_volume"])] = 1

for i, key in enumerate(all_masks_indexed_hdf5_z_brain10.keys()):
    masks_zbrain10[i][tuple(all_masks_indexed_hdf5_z_brain10[key]["ind_mask_volume"])] = 1
print("... Done")


fp = open("/Users/arminbahl/Desktop/test.csv", "w")
fp.write("x,y,z,mapzebrain,zbrain10\n")

for z_plane in range(6):
    unit_centroids_ants_registered = container_functional_hdf5[f"{z_plane}/unit_centroids_ants_registered"]

    for x, y, z in unit_centroids_ants_registered:
        
        regions_mapzebrain = ""
        for i, key in enumerate(all_masks_indexed_hdf5_mapzebrain.keys()):
            if masks_mapzebrain[i, round(z), round(y), round(x)] == 1:
                regions_mapzebrain += key
                regions_mapzebrain += ','

        if len(regions_mapzebrain) > 0:
            regions_mapzebrain = regions_mapzebrain[:-1]
            
        regions_zbrain10 = ""
        for i, key in enumerate(all_masks_indexed_hdf5_z_brain10.keys()):
            if masks_zbrain10[i, round(z), round(y), round(x)] == 1:
                regions_zbrain10 += key
                regions_zbrain10 += ','

        if len(regions_zbrain10) > 0:
            regions_zbrain10 = regions_zbrain10[:-1]
            
        

        fp.write(f'{x:.2f},{y:.2f},{z:.2f},"{regions_mapzebrain}","{regions_zbrain10}"\n')

fp.close()


