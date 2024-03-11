import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import h5py


for item in os.listdir(r'W:\Florian\function-neurotransmitter-morphology\functional'):
    try:
        with h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\functional\{item}\{item}_preprocessed_data.h5") as f:

            plt.figure()
            image = f['average_stack_green_channel'][0, :, :]
            image = np.clip(image,np.percentile(image,2),np.percentile(image,98))
            plt.imshow(image, 'gray')
            try:
                plt.plot(f['z_plane0000/manual_segmentation/unit_contours/0'][:, 0], f['z_plane0000/manual_segmentation/unit_contours/0'][:, 1], c='red', alpha=1)
            except:
                plt.plot(f['z_plane0000/manual_segmentation/unit_contours/10000'][:, 0], f['z_plane0000/manual_segmentation/unit_contours/10000'][:, 1], c='red', alpha=1)
            plt.text((plt.gca().get_xlim()[0] + plt.gca().get_xlim()[1]) / 2, (plt.gca().get_ylim()[0] + plt.gca().get_ylim()[1]) / 2, item, ha='center', va='center', c='white')
            plt.savefig(rf'W:\Florian\function-neurotransmitter-morphology\functional\all_cell_rois\{item}.png')



    except Exception as e:
        print(f"{item.upper()}")
        print(f"An error occurred: {e}\n")

