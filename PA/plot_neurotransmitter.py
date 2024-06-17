import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from cellpose import denoise
from cellpose import denoise
import cv2
import pandas as pd
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
from hindbrain_structure_function.visualization.FK_tools.load_pa_table import *
import re



def plot_hcr(gad1b_path,vglut2a_path,path_to_data,cell_name,zoomfactor = 20,roi=True,pdf=False):
    def merge_gad_vglut_green_pink(gad, vglut):
        gad = cv2.normalize(gad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vglut = cv2.normalize(vglut, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        merged = cv2.merge([vglut, gad, vglut])
        return merged

    def merge_gad_vglut_red_blue(gad, vglut):
        gad = cv2.normalize(gad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vglut = cv2.normalize(vglut, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        green_channel = np.zeros_like(gad)
        merged = cv2.merge([vglut, green_channel, gad])
        return merged

    def merge_gad_vglut_gcamp(gad, vglut, gcamp):
        gad = cv2.normalize(gad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vglut = cv2.normalize(vglut, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gcamp = cv2.normalize(gcamp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        merged = cv2.merge([vglut, gcamp, gad])
        return merged
    if (gad1b_path    / (gad1b_path.name      + "_roi.tiff")).exists():
        gad1b_roi       = tiff.imread(gad1b_path    / (gad1b_path.name      + "_roi.tiff"))
    else:
        gad1b_roi       = tiff.imread(gad1b_path    / (gad1b_path.name      + "_roi.tif"))

    if (vglut2a_path  / (vglut2a_path.name    + "_roi.tiff")).exists():
        vglut2a_roi       = tiff.imread(vglut2a_path    / (vglut2a_path.name      + "_roi.tiff"))
    else:
        vglut2a_roi       = tiff.imread(vglut2a_path    / (vglut2a_path.name      + "_roi.tif"))




    if (vglut2a_path / (vglut2a_path.name + "_wv-950_preprocessed_data.h5")).exists():
        gad1b_gcamp = np.array(h5py.File(gad1b_path / (gad1b_path.name + "_preprocessed_data.h5"))['average_stack_green_channel'])
        gad1b_signal = np.array(h5py.File(gad1b_path / (gad1b_path.name + "_preprocessed_data.h5"))['average_stack_red_channel'])
        vglut2a_gcamp = np.array(h5py.File(vglut2a_path / (vglut2a_path.name + "_wv-950_preprocessed_data.h5"))['average_stack_red_channel'])
        vglut2a_signal   = np.array(h5py.File(vglut2a_path    / (vglut2a_path.name      + "_wv-800_preprocessed_data.h5"))['average_stack_green_channel'])
    elif 'FK' not in str(gad1b_path):
        gad1b_gcamp = np.array(h5py.File(gad1b_path / (gad1b_path.name + "_preprocessed_data.h5"))['average_stack_green_channel'])
        gad1b_signal = np.array(h5py.File(gad1b_path / (gad1b_path.name + "_preprocessed_data.h5"))['average_stack_red_channel'])
        vglut2a_gcamp = np.array(h5py.File(vglut2a_path / (vglut2a_path.name + "_preprocessed_data.h5"))['average_stack_red_channel'])
        vglut2a_signal   = np.array(h5py.File(vglut2a_path / (vglut2a_path.name + "_preprocessed_data.h5"))['average_stack_green_channel'])
    elif 'FK' in str(gad1b_path):
        gad1b_gcamp = np.array(h5py.File(gad1b_path / (gad1b_path.name + "_preprocessed_data.h5"))['average_stack_repeat00_tile000_1020nm_channel0'])
        gad1b_signal = np.array(h5py.File(gad1b_path / (gad1b_path.name + "_preprocessed_data.h5"))['average_stack_repeat00_tile000_1020nm_channel1'])
        vglut2a_gcamp = np.array(h5py.File(vglut2a_path / (vglut2a_path.name + "_preprocessed_data.h5"))['average_stack_repeat00_tile000_950nm_channel1'])
        vglut2a_signal   = np.array(h5py.File(vglut2a_path / (vglut2a_path.name + "_preprocessed_data.h5"))['average_stack_repeat00_tile000_800nm_channel0'])







    #subset to rois

    gad1b_roi_coords = np.where(gad1b_roi==0)
    vglut2a_roi_coords = np.where(vglut2a_roi==0)

    gad1b_roi_overlay =  np.empty(gad1b_roi.shape)
    vglut2a_roi_overlay =  np.empty(vglut2a_roi.shape)

    gad1b_roi_overlay[gad1b_roi_coords[0],gad1b_roi_coords[1],gad1b_roi_coords[2]] = 1
    vglut2a_roi_overlay[vglut2a_roi_coords[0],vglut2a_roi_coords[1],vglut2a_roi_coords[2]] = 1

    def calc_limits():

        gad1b_roi_xmin = np.min(gad1b_roi_coords[1])
        gad1b_roi_xmax = np.max(gad1b_roi_coords[1])
        gad1b_roi_ymin = np.min(gad1b_roi_coords[2])
        gad1b_roi_ymax = np.max(gad1b_roi_coords[2])
        gad1b_roi_xy_center = [gad1b_roi_xmax - (gad1b_roi_xmax-gad1b_roi_xmin)/2,gad1b_roi_ymax - (gad1b_roi_ymax-gad1b_roi_ymin)/2]
        gad1b_roi_xy_center = [np.floor(x) for x in gad1b_roi_xy_center]

        vglut2a_roi_xmin = np.min(vglut2a_roi_coords[1])
        vglut2a_roi_xmax = np.max(vglut2a_roi_coords[1])
        vglut2a_roi_ymin = np.min(vglut2a_roi_coords[2])
        vglut2a_roi_ymax = np.max(vglut2a_roi_coords[2])
        vglut2a_roi_xy_center = [vglut2a_roi_xmax - (vglut2a_roi_xmax-vglut2a_roi_xmin)/2,vglut2a_roi_ymax - (vglut2a_roi_ymax-vglut2a_roi_ymin)/2]
        vglut2a_roi_xy_center = [np.floor(x) for x in vglut2a_roi_xy_center]

        dist_from_center = np.ceil(np.max([vglut2a_roi_xmax-vglut2a_roi_xy_center[0],vglut2a_roi_xy_center[0]-vglut2a_roi_xmin,
                                  gad1b_roi_xmax-gad1b_roi_xy_center[0],gad1b_roi_xy_center[0]-gad1b_roi_xmin,
                                  vglut2a_roi_ymax - vglut2a_roi_xy_center[1], vglut2a_roi_xy_center[1] - vglut2a_roi_ymin,
                                  gad1b_roi_ymax - gad1b_roi_xy_center[1], gad1b_roi_xy_center[1] - gad1b_roi_ymin],
                                  ) * zoomfactor)

        gad1b_xmin = int(gad1b_roi_xy_center[0] - dist_from_center)
        gad1b_xmax = int(gad1b_roi_xy_center[0] + dist_from_center)
        gad1b_ymin = int(gad1b_roi_xy_center[1] - dist_from_center)
        gad1b_ymax = int(gad1b_roi_xy_center[1] + dist_from_center)

        vglut2a_xmin = int(vglut2a_roi_xy_center[0] - dist_from_center)
        vglut2a_xmax = int(vglut2a_roi_xy_center[0] + dist_from_center)
        vglut2a_ymin = int(vglut2a_roi_xy_center[1] - dist_from_center)
        vglut2a_ymax = int(vglut2a_roi_xy_center[1] + dist_from_center)
        vglut2a_limits = [vglut2a_xmin, vglut2a_roi_ymin, vglut2a_xmax, vglut2a_ymax]
        gad1b_limits = [gad1b_xmin, gad1b_roi_ymin, gad1b_xmax, gad1b_ymax]

        while (abs(gad1b_xmax)-abs(gad1b_xmin) != abs(vglut2a_xmax)-abs(vglut2a_xmin))|(abs(gad1b_ymax)-abs(gad1b_ymin) != abs(vglut2a_ymax)-abs(vglut2a_ymin)):


            for limit, coord in zip(gad1b_limits+ vglut2a_limits,(gad1b_roi_xy_center*2)+(vglut2a_roi_xy_center*2)):
                if limit < 0:
                    dist_from_center = coord


                if limit > 799:
                    dist_from_center = 799-coord


            gad1b_xmin = int(gad1b_roi_xy_center[0] - dist_from_center)
            gad1b_xmax = int(gad1b_roi_xy_center[0] + dist_from_center)
            gad1b_ymin = int(gad1b_roi_xy_center[1] - dist_from_center)
            gad1b_ymax = int(gad1b_roi_xy_center[1] + dist_from_center)

            vglut2a_xmin = int(vglut2a_roi_xy_center[0] - dist_from_center)
            vglut2a_xmax = int(vglut2a_roi_xy_center[0] + dist_from_center)
            vglut2a_ymin = int(vglut2a_roi_xy_center[1] - dist_from_center)
            vglut2a_ymax = int(vglut2a_roi_xy_center[1] + dist_from_center)

            vglut2a_limits = [vglut2a_xmin, vglut2a_ymin, vglut2a_xmax, vglut2a_ymax]
            gad1b_limits = [gad1b_xmin, gad1b_ymin, gad1b_xmax, gad1b_ymax]
        gad1b_xmin = int(gad1b_roi_xy_center[0] - dist_from_center)
        gad1b_xmax = int(gad1b_roi_xy_center[0] + dist_from_center)
        gad1b_ymin = int(gad1b_roi_xy_center[1] - dist_from_center)
        gad1b_ymax = int(gad1b_roi_xy_center[1] + dist_from_center)

        vglut2a_xmin = int(vglut2a_roi_xy_center[0] - dist_from_center)
        vglut2a_xmax = int(vglut2a_roi_xy_center[0] + dist_from_center)
        vglut2a_ymin = int(vglut2a_roi_xy_center[1] - dist_from_center)
        vglut2a_ymax = int(vglut2a_roi_xy_center[1] + dist_from_center)
        vglut2a_limits = [vglut2a_xmin, vglut2a_ymin, vglut2a_xmax, vglut2a_ymax]
        gad1b_limits = [gad1b_xmin, gad1b_ymin, gad1b_xmax, gad1b_ymax]


        return gad1b_limits,vglut2a_limits

    gad1b_limits,vglut2a_limits = calc_limits()

    gad1b_xmin,gad1b_ymin,gad1b_xmax,gad1b_ymax = gad1b_limits
    vglut2a_xmin, vglut2a_ymin, vglut2a_xmax, vglut2a_ymax = vglut2a_limits
    
    gad1b_gcamp = gad1b_gcamp[np.unique(gad1b_roi_coords[0]), gad1b_xmin:gad1b_xmax, gad1b_ymin:gad1b_ymax]
    vglut2a_gcamp = vglut2a_gcamp[np.unique(vglut2a_roi_coords[0]), vglut2a_xmin:vglut2a_xmax, vglut2a_ymin:vglut2a_ymax]

    gad1b_signal = gad1b_signal[np.unique(gad1b_roi_coords[0]), gad1b_xmin:gad1b_xmax, gad1b_ymin:gad1b_ymax]
    vglut2a_signal = vglut2a_signal[np.unique(vglut2a_roi_coords[0]), vglut2a_xmin:vglut2a_xmax, vglut2a_ymin:vglut2a_ymax]

    gad1b_roi_overlay = gad1b_roi_overlay[np.unique(gad1b_roi_coords[0]), gad1b_xmin:gad1b_xmax, gad1b_ymin:gad1b_ymax]
    vglut2a_roi_overlay = vglut2a_roi_overlay[np.unique(vglut2a_roi_coords[0]), vglut2a_xmin:vglut2a_xmax, vglut2a_ymin:vglut2a_ymax]




    #clip signal
    gad1b_gcamp_clipped =  np.clip(gad1b_gcamp,np.percentile(gad1b_gcamp,5),np.percentile(gad1b_gcamp,95))
    gad1b_signal_clipped = np.clip(gad1b_signal,np.percentile(gad1b_signal,5),np.percentile(gad1b_signal,95))
    vglut2a_gcamp_clipped =  np.clip(vglut2a_gcamp,np.percentile(vglut2a_gcamp,5),np.percentile(vglut2a_gcamp,95))
    vglut2a_signal_clipped = np.clip(vglut2a_signal,np.percentile(vglut2a_signal,5),np.percentile(vglut2a_signal,95))
    #make target folder
    os.makedirs(path_to_data / 'make_figures_FK_output' / 'hcr_plots'/cell_name,exist_ok=True)
    output_folder = path_to_data / 'make_figures_FK_output' / 'hcr_plots'/cell_name

    #plot without cells
    # plt.imshow(np.mean(gad1b_gcamp_clipped,axis=0),'gray')
    plt.imshow(np.max(gad1b_signal_clipped,axis=0),'Reds',alpha=1)
    if roi:
        plt.pcolormesh(np.max(gad1b_roi_overlay,axis=0),alpha =np.max(gad1b_roi_overlay,axis=0)*0.2)
    plt.title('Gad1b')
    plt.savefig(output_folder/'0.png')
    if pdf:
        plt.savefig(output_folder / '0.pdf')


    plt.imshow(np.max(vglut2a_signal_clipped,axis=0),'Blues',alpha=1)
    if roi:
        plt.pcolormesh(np.max(vglut2a_roi_overlay,axis=0),alpha =np.max(vglut2a_roi_overlay,axis=0)*0.1)
    plt.title('Vglut2a')
    plt.savefig(output_folder/'1.png')
    if pdf:
        plt.savefig(output_folder / '1.pdf')
    plt.close(plt.gcf())

    #plot with cells
    plt.imshow(np.mean(gad1b_gcamp_clipped,axis=0),'gray')
    plt.imshow(np.max(gad1b_signal_clipped,axis=0),'Reds',alpha=0.5)
    if roi:
        plt.pcolormesh(np.max(gad1b_roi_overlay,axis=0),alpha =np.max(gad1b_roi_overlay,axis=0)*0.4)
    plt.title('Gad1b witch cell bg')
    plt.savefig(output_folder/'2.png')
    if pdf:
        plt.savefig(output_folder / '2.pdf')
    plt.close(plt.gcf())

    plt.imshow(np.mean(vglut2a_gcamp_clipped,axis=0),'gray')
    plt.imshow(np.max(vglut2a_signal_clipped,axis=0),'Blues',alpha=0.5)
    if roi:
        plt.pcolormesh(np.max(vglut2a_roi_overlay,axis=0),alpha =np.max(vglut2a_roi_overlay,axis=0)*0.4)
    plt.title('Vglut2a')
    plt.savefig(output_folder/'3.png')
    if pdf:
        plt.savefig(output_folder / '3.pdf')
    plt.close(plt.gcf())


    #plot per plane

    fig,ax = plt.subplots(1,gad1b_signal_clipped.shape[0],figsize=(30,20))
    for i in range(gad1b_signal_clipped.shape[0]):
        try:
            ax[i].imshow(gad1b_signal_clipped[i,:,:], 'Reds', alpha=1)
            if roi:
                ax[i].pcolormesh(gad1b_roi_overlay[i,:,:], alpha=gad1b_roi_overlay[i,:,:] * 0.4)
        except:
            ax.imshow(gad1b_signal_clipped[i, :, :], 'Reds', alpha=1)
            if roi:
                ax.pcolormesh(gad1b_roi_overlay[i, :, :], alpha=gad1b_roi_overlay[i, :, :] * 0.4)
    plt.savefig(output_folder/'4.png')
    if pdf:
        plt.savefig(output_folder / '4.pdf')
    plt.close(plt.gcf())
    fig,ax = plt.subplots(1,vglut2a_signal_clipped.shape[0],figsize=(30,20))
    for i in range(vglut2a_signal_clipped.shape[0]):
        try:
            ax[i].imshow(vglut2a_signal_clipped[i,:,:], 'Blues', alpha=1)
            if roi:
                ax[i].pcolormesh(vglut2a_roi_overlay[i,:,:], alpha=vglut2a_roi_overlay[i,:,:] * 0.4)
        except:
            ax.imshow(vglut2a_signal_clipped[i,:,:], 'Blues', alpha=1)
            if roi:
                ax.pcolormesh(vglut2a_roi_overlay[i,:,:], alpha=vglut2a_roi_overlay[i,:,:] * 0.4)
    plt.savefig(output_folder/'5.png')
    if pdf:
        plt.savefig(output_folder / '5.pdf')
    plt.close(plt.gcf())

    normed_gad1b = gad1b_signal_clipped/np.max(gad1b_signal_clipped)
    normed_vglut2a = vglut2a_signal_clipped/np.max(vglut2a_signal_clipped)
    # plt.imshow(merge_gad_vglut_gcamp(np.mean(normed_vglut2a,axis=0),np.mean(normed_gad1b,axis=0),np.mean(gad1b_gcamp_clipped,axis=0)))
    plt.imshow(np.mean(gad1b_gcamp_clipped, axis=0),'gray')
    if roi:
        plt.pcolormesh(np.max(vglut2a_roi_overlay,axis=0),alpha =np.max(vglut2a_roi_overlay,axis=0)*0.2)
    plt.savefig(output_folder/'6.png')
    if pdf:
        plt.savefig(output_folder / '6.pdf')
    plt.close(plt.gcf())


    plt.imshow(merge_gad_vglut_red_blue(np.mean(normed_vglut2a,axis=0),np.mean(normed_gad1b,axis=0)))
    if roi:
        plt.pcolormesh(np.max(vglut2a_roi_overlay,axis=0),alpha =np.max(vglut2a_roi_overlay,axis=0)*0.2)
    plt.savefig(output_folder/'7.png')
    if pdf:
        plt.savefig(output_folder / '7.pdf')
    plt.close(plt.gcf())

    plt.imshow(merge_gad_vglut_green_pink(np.mean(normed_vglut2a,axis=0),np.mean(normed_gad1b,axis=0)))
    if roi:
        plt.pcolormesh(np.max(vglut2a_roi_overlay,axis=0),alpha =np.max(vglut2a_roi_overlay,axis=0)*0.2)
    plt.savefig(output_folder/'8.png')
    if pdf:
        plt.savefig(output_folder / '8.pdf')
    print(cell_name, 'finished')
    plt.close(plt.gcf())


path_to_data = get_base_path()
pa_table = load_pa_table(path_to_data.joinpath("paGFP").joinpath("photoactivation_cells_table.csv"))


def contains_pattern(s):
    # Define the regular expression pattern
    pattern = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'

    # Use re.search to check if any substring matches the pattern
    if type(s) == str:
        match = re.search(pattern, s)
        return bool(match)
    else:
        return False


for i,cell in pa_table.iterrows():







    if contains_pattern(cell['gad1b_ID']):
        gad1b_path     =  Path(rf'W:\Florian\function-neurotransmitter-morphology\HCR\{cell.gad1b_ID}')
        vglut2a_path   = Path(rf'W:\Florian\function-neurotransmitter-morphology\HCR\{cell.vglut2a_ID}')

        plot_hcr(gad1b_path,vglut2a_path,path_to_data=path_to_data,cell_name=cell.cell_name)


#example good cell
cell = pa_table.loc[pa_table['cell_name']=='20230403.1',:].iloc[0]
gad1b_path = Path(rf'W:\Florian\function-neurotransmitter-morphology\HCR\{cell.gad1b_ID}')
vglut2a_path = Path(rf'W:\Florian\function-neurotransmitter-morphology\HCR\{cell.vglut2a_ID}')

plot_hcr(gad1b_path, vglut2a_path, path_to_data=path_to_data, cell_name=cell.cell_name,zoomfactor = 20,pdf=True)