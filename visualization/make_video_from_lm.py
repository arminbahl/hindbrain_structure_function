import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import os
from PIL import Image, ImageDraw, ImageFont
import shutil
from tqdm import tqdm
import glob
import subprocess

path_to_functional = Path(r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\ipsilateral_integrator for gregor\functional\2023-05-26_11-00-06_raw_data.hdf5")
dpi = 300
functional_file = h5py.File(path_to_functional, "r")
functional_data = functional_file['z_plane0000/imaging_green_channel']
functional_data = np.clip(functional_data,np.percentile(functional_data,2),np.percentile(functional_data,98))



os.makedirs('temp', exist_ok=True)

frames_filenames = []
frames=[]
for i in tqdm(range(functional_data.shape[0]),leave=False):
    frame_filename = rf"temp\frame{i}.jpg"
    frames_filenames.append(frame_filename)

    plt.imshow(functional_data[i,:,:],'gray')
    plt.axis('off')
    plt.savefig(frame_filename, dpi=dpi)




for i in tqdm(range(functional_data.shape[0]),leave=False):
    temp_image = np.array(Image.open(frame_filename))
    frames.append(temp_image)
shutil.rmtree('temp')
imageio.mimsave(Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\ipsilateral_integrator for gregor').joinpath('20230526.2.mp4'), frames, fps=30,
                codec="libx264", output_params=["-crf", "20"])