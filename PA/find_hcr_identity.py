from PIL import Image, ImageDraw, ImageFont
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from copy import deepcopy
size_window = 100
import cv2

#generate missing image
image = np.empty(shape=(800,800,3))
missing =cv2.putText(img=np.copy(image), text="hello", org=(200,200),fontFace=3, fontScale=3, color=(0,0,255), thickness=5)




#load cell registers

path_to_nextcloud = Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data')
os.makedirs(path_to_nextcloud.joinpath('make_figures_FK_output').joinpath('neurotransmitter').joinpath('HCR_evaluation'),exist_ok=True)
path_to_output = path_to_nextcloud.joinpath('make_figures_FK_output').joinpath('neurotransmitter').joinpath('HCR_evaluation')

cell_register = pd.read_csv(r"C:\Users\ag-bahl\Desktop\cell_register\cell_register_translation.csv")
cell_register_full = pd.read_csv(r"C:\Users\ag-bahl\Desktop\cell_register\cell_register_full.csv")

#remove artifacts
for column in cell_register.columns:
    if column != 'cell_type_labels':
        cell_register[column] = cell_register[column].apply(lambda x: x.replace('"', '') if type(x) == str else x)

cell_register_full['Internal name'] = cell_register_full['Internal name'].astype(str)


#subset_cell registers

cell_register = cell_register.loc[~(cell_register['vglut2a_ID'].isna())&~(cell_register['gad1b_ID'].isna()),:]
subset_list = [x in list(cell_register['cell_name']) for x in cell_register_full['Internal name']]
cell_register_full = cell_register_full.loc[subset_list,:]
cell_register_full = cell_register_full.loc[~cell_register_full['Location cell in HCR 1020'].isna(),:]
size_window = size_window/2

for i,cell in cell_register_full.iterrows():
    gcamp_HCR800 = deepcopy(missing)
    HCR_HCR800 = deepcopy(missing)
    gcamp_HCR1020 = deepcopy(missing)
    HCR_HCR1020 = deepcopy(missing)


    HCR1020_location  = cell_register.loc[cell_register['cell_name'] == cell['Internal name'],'gad1b_ID'].iloc[0]
    HCR800_location = cell_register.loc[cell_register['cell_name'] == cell['Internal name'], 'vglut2a_ID'].iloc[0]

    x_HCR1020, y_HCR1020, z_HCR1020 = [int(x) for x in cell['Location cell in HCR 1020'].replace(" ","").split(",")]
    x_HCR800, y_HCR800, z_HCR800 = [int(x) for x in cell['Location cell in HCR 800'].replace(" ", "").split(",")]

    # get HCR1020 
    f =  h5py.File(rf'W:\Florian\function-neurotransmitter-morphology\HCR\{HCR1020_location}\{HCR1020_location}_preprocessed_data.h5')
    x_low = int(x_HCR1020-size_window)
    x_high = int(x_HCR1020+size_window)

    y_low = int(y_HCR1020 - size_window)
    y_high = int(y_HCR1020 + size_window)


    if 'average_stack_green_channel' in f.keys():
        gcamp_HCR1020 = np.array(f['average_stack_green_channel'][z_HCR1020,x_low:x_high,y_low:y_high])
        HCR_HCR1020 = np.array(f['average_stack_red_channel'][z_HCR1020,x_low:x_high,y_low:y_high])

    if 'average_stack_repeat00_tile000_1020nm_channel0' in f.keys():
        gcamp_HCR1020 = np.array(f['average_stack_repeat00_tile000_1020nm_channel0'][z_HCR1020,x_low:x_high,y_low:y_high])
        HCR_HCR1020 = np.array(f['average_stack_repeat00_tile000_1020nm_channel1'][z_HCR1020,x_low:x_high,y_low:y_high])

    gcamp_HCR1020 = np.clip(gcamp_HCR1020, np.percentile(gcamp_HCR1020, 1), np.percentile(gcamp_HCR1020, 99))
    HCR_HCR1020 = np.clip(HCR_HCR1020, np.percentile(HCR_HCR1020, 1), np.percentile(HCR_HCR1020, 99))
    f.close()
    
    
    #get HCR800
    if Path(rf'W:\Florian\function-neurotransmitter-morphology\HCR\{HCR800_location}\{HCR800_location}_wv-800_preprocessed_data.h5').exists():
        f_HCR = h5py.File(rf'W:\Florian\function-neurotransmitter-morphology\HCR\{HCR800_location}\{HCR800_location}_wv-800_preprocessed_data.h5')
        f_GCAMP = h5py.File(rf'W:\Florian\function-neurotransmitter-morphology\HCR\{HCR800_location}\{HCR800_location}_wv-950_preprocessed_data.h5')
        flag_split_wv = True
    else:
        f = h5py.File(rf'W:\Florian\function-neurotransmitter-morphology\HCR\{HCR800_location}\{HCR800_location}_preprocessed_data.h5')
        flag_split_wv = False

    x_low = int(x_HCR800 - size_window)
    x_high = int(x_HCR800 + size_window)

    y_low = int(y_HCR800 - size_window)
    y_high = int(y_HCR800 + size_window)

    if flag_split_wv:
        if 'average_stack_green_channel' in f_HCR.keys():
            gcamp_HCR800 = np.array(f_GCAMP['average_stack_red_channel'][z_HCR800, x_low:x_high, y_low:y_high])
            HCR_HCR800 = np.array(f_HCR['average_stack_green_channel'][z_HCR800, x_low:x_high, y_low:y_high])

    elif not flag_split_wv:
            if 'average_stack_green_channel' in f.keys():
                gcamp_HCR800 = np.array(f['average_stack_green_channel'][z_HCR800, x_low:x_high, y_low:y_high])
                HCR_HCR800 = np.array(f['average_stack_red_channel'][z_HCR800, x_low:x_high, y_low:y_high])

            if 'average_stack_repeat00_tile000_1020nm_channel0' in f.keys():
                gcamp_HCR800 = np.array(f['average_stack_repeat00_tile000_800nm_channel0'][z_HCR800, x_low:x_high, y_low:y_high])
                HCR_HCR800 = np.array(f['average_stack_repeat00_tile000_800nm_channel1'][z_HCR800, x_low:x_high, y_low:y_high])




    


    gcamp_HCR800 = np.clip(gcamp_HCR800, np.percentile(gcamp_HCR800, 1), np.percentile(gcamp_HCR800, 99))
    HCR_HCR800 = np.clip(HCR_HCR800, np.percentile(HCR_HCR800, 1), np.percentile(HCR_HCR800, 99))
    f.close()









    #plotting part
    fig,ax = plt.subplots(2,2,dpi=300)
    plt.suptitle(cell['Internal name'])
    ax[0,0].imshow(gcamp_HCR1020,cmap='gray')
    ax[0, 0].scatter(size_window,size_window,marker='x',c='r',alpha=0.5)
    ax[0, 1].imshow(HCR_HCR1020,cmap='gray')
    ax[0, 1].scatter(size_window, size_window, marker='x', c='r',alpha=0.5)
    ax[1, 0].imshow(gcamp_HCR800, cmap='gray')
    ax[1, 0].scatter(size_window, size_window, marker='x', c='r',alpha=0.5)
    ax[1, 1].imshow(HCR_HCR800, cmap='gray')
    ax[1, 1].scatter(size_window, size_window, marker='x', c='r',alpha=0.5)
    plt.axis('off')
    for i in ax:
        for i2 in i:
            i2.axis('off')

    plt.savefig(path_to_output.joinpath(cell['Internal name']+".png"))


