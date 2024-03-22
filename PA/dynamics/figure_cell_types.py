import matplotlib.pyplot as plt
import numpy as np
from analysis_helpers.analysis.personal_dirs.Florian.thesis_plot.Morphologies.utils.extract_traces_from_roi import *
from skimage import exposure
import pandas as pd
import datetime
import cv2
from pathlib import Path
import h5py
from scipy.signal import savgol_filter
#custom functions

def calculate_dynamic(functional_name):
    with h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\functional\{functional_name}\{functional_name}_preprocessed_data.h5",'r') as f:
        stimulus_aligned_F0 = np.array(f["z_plane0000/manual_segmentation/stimulus_aligned_dynamics/stimulus0000/F"])
        stimulus_aligned_F1 = np.array(f["z_plane0000/manual_segmentation/stimulus_aligned_dynamics/stimulus0001/F"])

    if np.nanmin([np.nanmin(stimulus_aligned_F0), np.nanmin(stimulus_aligned_F1)])<0:
        stimulus_aligned_F0 = stimulus_aligned_F0 + np.abs(
            np.nanmin([np.nanmin(stimulus_aligned_F0), np.nanmin(stimulus_aligned_F1)]))
        stimulus_aligned_F1 = stimulus_aligned_F1 + np.abs(
            np.nanmin([np.nanmin(stimulus_aligned_F0), np.nanmin(stimulus_aligned_F1)]))

    df_f0_stim0 = (stimulus_aligned_F0[:, 0, :] - np.nanmean(stimulus_aligned_F0[:, 0, 0:20], axis=1)[:,
                                                  np.newaxis]) / np.nanmean(stimulus_aligned_F0[:, 0, 0:20], axis=1)[:,
                                                                 np.newaxis]
    df_f0_stim1 = (stimulus_aligned_F1[:, 0, :] - np.nanmean(stimulus_aligned_F1[:, 0, 0:20], axis=1)[:,
                                                  np.newaxis]) / np.nanmean(stimulus_aligned_F1[:, 0, 0:20], axis=1)[:,
                                                                 np.newaxis]

    average_dynamic0 = np.mean(df_f0_stim0, axis=0)
    average_dynamic1 = np.mean(df_f0_stim1, axis=0)

    return np.array(average_dynamic0),df_f0_stim0, np.array(average_dynamic1), df_f0_stim1
clip_percentile = 1.2
def get_hcr_image(hcr_gad_name,hcr_vglut_name, z_gad,z_vglut):
    global clip_percentile
    with h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\HCR\{hcr_gad_name}\{hcr_gad_name}_preprocessed_data.h5", 'r') as f:
        gad_gcamp = np.array(f["average_stack_green_channel"][z_gad[-1], :, :])
        gad_gcamp = np.clip(gad_gcamp, np.percentile(gad_gcamp, clip_percentile), np.percentile(gad_gcamp, 100-clip_percentile))

        gad_hcr = np.array(f["average_stack_red_channel"][z_gad[-1], :, :])
        gad_hcr = np.clip(gad_hcr, np.percentile(gad_hcr, clip_percentile), np.percentile(gad_hcr, 100-clip_percentile))
    with h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\HCR\{hcr_vglut_name}\{hcr_vglut_name}_wv-800_preprocessed_data.h5", 'r') as f:
        vglut = np.array(f["average_stack_green_channel"][z_vglut[-1], :, :])
        vglut = np.clip(vglut, np.percentile(vglut, clip_percentile), np.percentile(vglut, 100-clip_percentile))

    return gad_gcamp,gad_hcr,vglut
def from_the_top(ypos,size_canvas=29.7):
    return size_canvas- ypos




dt_example = '2023-04-24_20-26-09'
motor_example = ''
ipsi_integrator_example = ''
contra_integrator_example = ''

path_to_cell_register = r"C:\Users\ag-bahl\Downloads\Table1.csv"
cell_register = pd.read_csv(path_to_cell_register)


# for cell in [dt_example,motor_example,ipsi_integrator_example, contra_integrator_example]:
# for cell in cell_register['Volume']:
for cell in ['2023-05-27_14-51-02']:
    try:
        temp_df = cell_register[cell_register["Volume"] == cell]
        path_to_swc_integrator = Path(rf'W:\Florian\function-neurotransmitter-morphology\{temp_df["Volume"].values[0]}\{temp_df["Volume"].values[0]+"-000.swc"}')
        functional = temp_df['Function'].iloc[0]


        #create figure
        main_fig = Figure()

        path_to_dynamics = Path(fr'W:\Florian\function-neurotransmitter-morphology\export\{cell}\{cell}_dynamics.hdf5')
        print(path_to_dynamics,path_to_dynamics.exists())
        #Dynamics
        with h5py.File(path_to_dynamics) as f:
            single_trials_left = np.array(f['dF_F/single_trial_rdms_left'])[:,0,:]
            single_trials_left = single_trials_left[~np.any((single_trials_left > np.nanpercentile(single_trials_left,99))|(single_trials_left < np.nanpercentile(single_trials_left,1)), axis=1),:]

            single_trials_right = np.array(f['dF_F/single_trial_rdms_right'])[:,0,:]
            single_trials_right = single_trials_right[~np.any((single_trials_right > np.nanpercentile(single_trials_right,99))|(single_trials_right < np.nanpercentile(single_trials_right,1)), axis=1),:]


            average_left = np.nanmean(single_trials_left,axis=0)
            average_right = np.nanmean(single_trials_right,axis=0)

        # Smooth using a Savitzky-Golay filter
        smooth_avg_activity_left = savgol_filter(average_left, 20, 3)
        smooth_avg_activity_right = savgol_filter(average_right, 20, 3)
        smooth_trials_left = savgol_filter(single_trials_left, 20, 3, axis=1)
        smooth_trials_right = savgol_filter(single_trials_right, 20, 3, axis=1)


        # ymin = (np.nanmin(np.hstack([smooth_trials_left,smooth_trials_right])))
        # ymax =(np.nanmax(np.hstack([smooth_trials_left,smooth_trials_right])))
        # ymin = np.round(ymin-0.2*ymax,1)
        # ymax = np.round(ymax+0.2*ymax,1)#rebound

        # Define time axis in seconds
        dt = 0.5  # Time step is 0.5 seconds
        time_axis = np.arange(len(smooth_avg_activity_left.flatten())) * dt
        # Plot smoothed average activity with thin lines
        fig, ax = plt.subplots()
        plt.plot(time_axis, smooth_avg_activity_left, color="blue", alpha=0.7, linewidth=3, label="Smoothed Average Left")
        plt.plot(time_axis, smooth_avg_activity_right, color="red", alpha=0.7, linestyle="--", linewidth=3, label="Smoothed Average Right")
        # Plot individual trial data with thin black lines for left and dashed black lines for right
        for trial_left, trial_right in zip(smooth_trials_left, smooth_trials_right):
                # if np.max(trial_left) <= 300:
                    plt.plot(time_axis, trial_left, color="black", alpha=0.3, linewidth=1)
                # if np.min(trial_right) >= -300:
                    plt.plot(time_axis, trial_right, color="black", alpha=0.3, linestyle="--", linewidth=1)
        # Overlay shaded rectangle for stimulus epoch
        plt.axvspan(10, 50, color="gray", alpha=0.1, label="Stimulus Epoch")
        plt.title(f"Average and Individual Trial Activity Dynamics for Neuron {temp_df['Internal name'].iloc[0]}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Activity")
        # Set font of legend text to Arial
        legend = plt.legend()
        for text in legend.get_texts():
            text.set_fontfamily("Arial")
        # Set aspect ratio to 1
        ratio = 1.0
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
        plt.savefig(rf"C:\Users\ag-bahl\Desktop\hindbrain_structure dynamics\{temp_df['Volume'].iloc[0]}.png")
        plt.show()

    except:
        pass




dynamics_figure = main_fig.create_plot(xpos=1,
                                            ypos=from_the_top(5),
                                            plot_height=4,plot_width=2,
                                            axis_off=False,plot_title='',plot_label="",
                                            ymin=ymin,ymax=ymax,xticks=[0,20,100,120],xticklabels=["0",'10','50',"60"],
                                            yticks=[0,np.round(np.nanmax([ipsi_rebound,contra_rebound]),1)],yticklabels=['0',str(np.round(np.nanmax([ipsi_rebound,contra_rebound]),1))])

show_rebound_dynamic_ipsi.ax.axvspan(20, 100, facecolor='gray', alpha=0.14)
show_rebound_dynamic_ipsi.ax.spines['right'].set_color('black')
show_rebound_dynamic_ipsi.ax.spines['top'].set_color('black')
show_rebound_dynamic_ipsi.ax.spines['top'].set_linewidth(0.5)
show_rebound_dynamic_ipsi.ax.spines['right'].set_linewidth(0.5)

show_rebound_dynamic_contra = main_fig.create_plot(xpos=3.25,
                                            ypos=from_the_top(5),
                                            plot_height=4,plot_width=2,
                                            axis_off=False,plot_title='',plot_label="",
                                            ymin=ymin,ymax=ymax,xticks=[0,20,100,120],xticklabels=["0",'10','50',"60"],
                                            yticks=[],yticklabels=[])

show_rebound_dynamic_contra.ax.axvspan(20, 100, facecolor='gray', alpha=0.14)
show_rebound_dynamic_contra.ax.spines['right'].set_color('black')
show_rebound_dynamic_contra.ax.spines['top'].set_color('black')
show_rebound_dynamic_contra.ax.spines['top'].set_linewidth(0.5)
show_rebound_dynamic_contra.ax.spines['right'].set_linewidth(0.5)

show_rebound_dynamic_ipsi.draw_line(np.arange(len(ipsi_rebound)),ipsi_rebound.astype(float))
for x in range(10):
    show_rebound_dynamic_ipsi.draw_line(np.arange(len(ipsi_rebound)), ipsi_all[x,:],lc='gray',alpha=0.2)

show_rebound_dynamic_contra.draw_line(np.arange(len(contra_rebound)), contra_rebound.astype(float))
for x in range(10):
    show_rebound_dynamic_contra.draw_line(np.arange(len(ipsi_rebound)), contra_all[x,:],lc='gray',alpha=0.2)

timestamp = datetime.datetime.now()
os.makedirs(fr"C:\Users\ag-bahl\Desktop\plot_diary\{timestamp.strftime('%Y_%m_%d')}", exist_ok=True)
main_fig.save(fr"C:\Users\ag-bahl\Desktop\plot_diary\{timestamp.strftime('%Y_%m_%d')}\{timestamp.strftime('%Y_%m_%d_%H_%M_%S_%f')}.{'pdf'}", open_file=True)

#HCR
#hcr plot
#integrator hcr
gad_name = integrator_df['1020 nm with Gad1b'].values[0]
vglut_name = integrator_df['800 nm with Vglut2a'].values[0].split(' ')[-1]
gad_location = [int(x) for x in integrator_df['Location cell in HCR 1020'].values[0].split(',')]
vglut_location = [int(x) for x in integrator_df['Location cell in HCR 800'].values[0].split(',')]

gad_gcamp_integrator,gad_integrator,vglut_integrator = get_hcr_image(gad_name,vglut_name,gad_location,vglut_location)

x_gad,y_gad = gad_location[:2]
x_vglut,y_vglut = vglut_location[:2]
imagesize = 100


size_square = [(imagesize/2)*0.7,(imagesize/2)*1.3]
xmin_gad, xmax_gad = x_gad-imagesize/2, x_gad+imagesize/2
ymin_gad, ymax_gad = y_gad-imagesize/2, y_gad+imagesize/2
xmin_vglut, xmax_vglut = x_vglut-imagesize/2, x_vglut+imagesize/2
ymin_vglut, ymax_vglut = y_vglut-imagesize/2, y_vglut+imagesize/2


show_gad_integrator = main_fig.create_plot(xpos=1,ypos=show_integrator_dynamic.plot_dict['ypos']-1-width_follow_neurites,
                             plot_height=width_follow_neurites,plot_width=width_follow_neurites,
                             axis_off=True,plot_title='',plot_label="d",xmin=0,xmax=imagesize,ymin=imagesize,ymax=0)

show_gad_integrator.draw_image(gad_integrator[int(ymin_gad):int(ymax_gad),int(xmin_gad):int(xmax_gad)],colormap='gray',extent=None)
show_gad_integrator.draw_polygon(y=[size_square[0],size_square[0],size_square[1],size_square[1],size_square[0]],
                               x=[size_square[0],size_square[1],size_square[1],size_square[0],size_square[0]],alpha=0,lc='red',lw=2)


show_vglut_integrator = main_fig.create_plot(xpos=follow_neurite_list[1].plot_dict['xpos'],ypos=show_gad_integrator.plot_dict['ypos'],
                             plot_height=width_follow_neurites,plot_width=width_follow_neurites,
                             axis_off=True,plot_title='',plot_label="",xmin=0,xmax=imagesize,ymin=imagesize,ymax=0)

show_vglut_integrator.draw_image(vglut_integrator[int(ymin_vglut):int(ymax_vglut),int(xmin_vglut):int(xmax_vglut)],colormap='gray',extent=None)
show_vglut_integrator.draw_polygon(y=[size_square[0],size_square[0],size_square[1],size_square[1],size_square[0]],
                               x=[size_square[0],size_square[1],size_square[1],size_square[0],size_square[0]],alpha=0,lc='red',lw=2)


show_gad_integrator = main_fig.create_plot(xpos=show_gad_integrator.plot_dict['xpos'],ypos=show_vglut_integrator.plot_dict['ypos']-distance_x-width_follow_neurites,
                             plot_height=width_follow_neurites,plot_width=width_follow_neurites,
                             axis_off=True,plot_title='',plot_label="",xmin=0,xmax=imagesize,ymin=imagesize,ymax=0)

show_gad_integrator.draw_image(gad_gcamp_integrator[int(ymin_gad):int(ymax_gad),int(xmin_gad):int(xmax_gad)],colormap='gray',extent=None)
show_gad_integrator.draw_polygon(y=[size_square[0],size_square[0],size_square[1],size_square[1],size_square[0]],
                               x=[size_square[0],size_square[1],size_square[1],size_square[0],size_square[0]],alpha=0,lc='red',lw=2)

show_composite_integrator = main_fig.create_plot(xpos=show_vglut_integrator.plot_dict['xpos'],ypos=show_vglut_integrator.plot_dict['ypos']-distance_x-width_follow_neurites,
                             plot_height=width_follow_neurites,plot_width=width_follow_neurites,
                             axis_off=True,plot_title='',plot_label="",xmin=0,xmax=imagesize,ymin=imagesize,ymax=0)

gad = cv2.normalize(gad_integrator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
vglut = cv2.normalize(vglut_integrator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
merged_gad_vglut_integrator = cv2.merge([vglut, gad,vglut])

show_composite_integrator.draw_image(merged_gad_vglut_integrator[int(ymin_gad):int(ymax_gad),int(xmin_gad):int(xmax_gad)],extent=None)
show_composite_integrator.draw_polygon(y=[size_square[0],size_square[0],size_square[1],size_square[1],size_square[0]],
                               x=[size_square[0],size_square[1],size_square[1],size_square[0],size_square[0]],alpha=0,lc='red',lw=2)











timestamp = datetime.datetime.now()
os.makedirs(fr"C:\Users\ag-bahl\Desktop\plot_diary\{timestamp.strftime('%Y_%m_%d')}", exist_ok=True)
main_fig.save(fr"C:\Users\ag-bahl\Desktop\plot_diary\{timestamp.strftime('%Y_%m_%d')}\{timestamp.strftime('%Y_%m_%d_%H_%M_%S_%f')}.{'pdf'}", open_file=True)
