import h5py
import pandas as pd
import numpy as np
import sys
import os
from matplotlib.lines import Line2D
sys.path.extend(['/Users/fkampf/PycharmProjects'])
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path

#Set path pointing at folder containing the data
path = Path('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud')
path_fish0 = path / 'clem_zfish1/activity_recordings/untitled folder/2025-03-05_13-12-33_fish000_setup0_arena0_AB_preprocessed_data.h5'
path_fish1 = path / 'clem_zfish1/activity_recordings/untitled folder/2025-03-05_13-12-40_fish001_setup1_arena0_AB_preprocessed_data.h5'
# Load regressors
regressors = np.load(path / 'make_figures_FK_output/functional_analysis/kmeans_regressors.npy')*100

# Load data from HDF5 files
with h5py.File(path_fish0) as f:
    F_fish0_rdms_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/left_left/F'])
    F_fish0_rdms_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/right_right/F'])
    F_fish0_no_motion = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/no_motion/F'])
    F_fish0_ramping_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/ramping_right/F'])
    F_fish0_ramping_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/ramping_left/F'])
    F_fish0_rdms_left_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/left_right/F'])
    F_fish0_rdms_right_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/right_left/F'])
    average_image_fish0 = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/imaging_data_channel0_time_averaged'])
    unit_names_fish0 = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_names'])
    hdf5_path_fish0 = np.full(unit_names_fish0.shape,fill_value=str(path_fish0))
    fish_id_fish0 = np.full(unit_names_fish0.shape,fill_value='fish0')
    fish0_df = pd.DataFrame([unit_names_fish0,hdf5_path_fish0,fish_id_fish0]).T
    fish0_df.columns = ['unit_name','hdf5_path','fish_id']

with h5py.File(path_fish1) as f:
    F_fish1_rdms_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/left_left/F'])
    F_fish1_rdms_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/right_right/F'])
    F_fish1_rdms_no_motion = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/no_motion/F'])
    F_fish1_ramping_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/ramping_right/F'])
    F_fish1_ramping_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/ramping_left/F'])
    F_fish1_rdms_left_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/left_right/F'])
    F_fish1_rdms_right_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/right_left/F'])
    average_image_fish1 = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/imaging_data_channel0_time_averaged'])
    unit_names_fish1 = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/unit_names'])
    hdf5_path_fish1 = np.full(unit_names_fish1.shape,fill_value=str(path_fish1))
    fish_id_fish1 = np.full(unit_names_fish1.shape,fill_value='fish1')
    fish1_df = pd.DataFrame([unit_names_fish1,hdf5_path_fish1,fish_id_fish1]).T
    fish1_df.columns = ['unit_name','hdf5_path','fish_id']


all_fish_df = pd.concat([fish0_df,fish1_df],axis=0)

# Combine data from both fish
F_rdms_left = np.concatenate([F_fish0_rdms_left,F_fish1_rdms_left], axis=1)[:,:,:120]
F_rdms_right = np.concatenate([F_fish0_rdms_right,F_fish1_rdms_right[:10,:,:]], axis=1)[:,:,:120]
F_no_motion = np.concatenate([F_fish0_no_motion, F_fish1_rdms_no_motion], axis=1)[:,:,:120]
F_ramping_right = np.concatenate([F_fish0_ramping_right, F_fish1_ramping_right], axis=1)[:,:,:120]
F_ramping_left = np.concatenate([F_fish0_ramping_left, F_fish1_ramping_left], axis=1)[:,:,:120]
F_rdms_left_right = np.concatenate([F_fish0_rdms_left_right, F_fish1_rdms_left_right], axis=1)[:,:,:120]
F_rdms_right_left = np.concatenate([F_fish0_rdms_right_left[:-1], F_fish1_rdms_right_left], axis=1)[:,:,:120]

def calc_dF_F(F, dt=0.5):
    """
    Calculate deltaF/F for given fluorescence data.
    
    Parameters:
    F (numpy array): Fluorescence data
    dt (float): Time step
    
    Returns:
    numpy array: deltaF/F data
    """
    F0 = np.nanmean(F[:, :, int(5 / dt):int(30 / dt)], axis=2, keepdims=True)
    dF_F = 100 * (F - F0) / F0
    return dF_F

def retrieve_segmentation(path_hdf5,id,seg_type='unit_contours'):
    """
    Retrieve segmentation data from an HDF5 file.

    Parameters:
    path_hdf5 (str): The file path to the HDF5 file containing the segmentation data.
    id (str): The identifier for the specific segmentation data to retrieve.
    seg_type (str, optional): The type of segmentation data to retrieve. Options are 'unit_contours', 
                                'unit_masks', and 'unit_contour_masks'. Default is 'unit_contours'.

    Returns:
    numpy.ndarray: The segmentation data as a NumPy array.

    Raises:
    KeyError: If the specified segmentation type or id does not exist in the HDF5 file.
    IOError: If there is an issue opening or reading the HDF5 file.
    """
    #options for seg_type are: unit_contours, unit_masks, unit_contour_masks
    with h5py.File(path_hdf5) as f:
        seg = np.array(f[f'repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/{seg_type}/{id}'])
    return seg

def plot_cells_in_brain(df, color_dict, cmap='gray', seg_type='unit_contours'):
    """
    Plots cells in the brain for each fish in the given DataFrame.
    Parameters:
    df (pd.DataFrame): DataFrame containing information about the cells. 
                       Must include columns 'fish_id', 'passes_cutoff', 'hdf5_path', 'unit_name', and 'functional_type'.
    color_dict (dict): Dictionary mapping functional types to colors.
    cmap (str, optional): Colormap to use for the background image. Default is 'gray'.
    seg_type (str, optional): Type of segmentation to retrieve. Default is 'unit_contours'.
    Returns:
    None: This function does not return any value. It displays a plot for each fish.
    """
   
    for fish in df['fish_id'].unique():
        passing_cells_per_fish = df[(df['fish_id']==fish)&(df['passes_cutoff']==True)]

        fig,ax = plt.subplots(1,1,figsize=(10,10))
        with h5py.File(passing_cells_per_fish['hdf5_path'].unique()[0]) as f:
            average_image = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/imaging_data_channel0_time_averaged'])
            average_image = np.clip(average_image,np.percentile(average_image,2),np.percentile(average_image,98))
        ax.imshow(average_image,cmap=cmap)

        for i,item in passing_cells_per_fish.iterrows():
            cell_seg = retrieve_segmentation(item['hdf5_path'],item['unit_name'],seg_type)
            
            ax.plot(cell_seg[:,0],cell_seg[:,1],color=color_dict[item['functional_type']],lw=1)

    plt.show()

def z_scorer(stim_activity_1, stim_activity_2, prestim, stim):
    """
    Calculate z-score between two stimulus activities.
    
    Parameters:
    stim_activity_1 (numpy array): First stimulus activity
    stim_activity_2 (numpy array): Second stimulus activity
    prestim (int): Pre-stimulus period
    stim (int): Stimulus period
    
    Returns:
    tuple: z-score up, z-score down, z-score
    """
    avg_activity_1 = np.nanmean(stim_activity_1, axis=0)
    avg_activity_2 = np.nanmean(stim_activity_2, axis=0)
    var_activity_1 = np.square(np.nanstd(stim_activity_1, axis=0)) / stim_activity_1.shape[0]
    var_activity_2 = np.square(np.nanstd(stim_activity_2, axis=0)) / stim_activity_2.shape[0]
    z_score = (avg_activity_1 - avg_activity_2) / np.sqrt(var_activity_1 + var_activity_2)
    z_score_up = np.nanpercentile(z_score[:, prestim:(prestim + stim)], 90, axis=1)  # Take the highest 10%
    z_score_down = np.nanpercentile(z_score[:, prestim:(prestim + stim)], 10, axis=1)
    return z_score_up, z_score_down

def reconcile_regressors(r0_all, r1_all, r2_all, r3_all, threshold=0.05):
    """
    Reconcile regressors by setting the maximum to its value and others to zero.
    If the difference between max and 2nd largest is less than threshold, set all to zero.
    
    Parameters:
    r0_all, r1_all, r2_all, r3_all (list): Lists of correlation coefficients
    threshold (float): Threshold for the difference between max and 2nd largest
    
    Returns:
    tuple: Reconciled lists of correlation coefficients
    """
    r0_all_reconciled = []
    r1_all_reconciled = []
    r2_all_reconciled = []
    r3_all_reconciled = []
    
    for r0, r1, r2, r3 in zip(r0_all, r1_all, r2_all, r3_all):
        values = [r0, r1, r2, r3]
        max_val = max(values)
        sorted_vals = sorted(values, reverse=True)
        
        if sorted_vals[0] - sorted_vals[1] < threshold:
            r0_all_reconciled.append(0)
            r1_all_reconciled.append(0)
            r2_all_reconciled.append(0)
            r3_all_reconciled.append(0)
        else:
            r0_all_reconciled.append(r0 if r0 == max_val else 0)
            r1_all_reconciled.append(r1 if r1 == max_val else 0)
            r2_all_reconciled.append(r2 if r2 == max_val else 0)
            r3_all_reconciled.append(r3 if r3 == max_val else 0)
    
    return r0_all_reconciled, r1_all_reconciled, r2_all_reconciled, r3_all_reconciled

def plot_dF_F_responses(dF_F_mean, 
                        r_left_list, r_right_list,
                        r_left_cutoff, r_right_cutoff,
                        z_left,z_right, 
                        title, legend_labels,z_limit=3.3):
    """
    Plot dF/F responses for given data.
    
    Parameters:
    dF_F_mean (numpy array): Mean dF/F data
    r_left_list (list of numpy arrays): List of correlation coefficients for left responses
    r_right_list (list of numpy arrays): List of correlation coefficients for right responses
    title (str): Title of the plot
    legend_labels (list): Labels for the legend
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(title)
    
    for i in range(4):
        r_left = r_left_list[i]
        r_right = r_right_list[i]
        ax[i//2, i%2].plot(dF_F_mean[(np.array(r_left) > r_left_cutoff[i])&(z_left>z_limit), :].T, c='green', lw=1, alpha=0.4)
        ax[i//2, i%2].plot(np.mean(dF_F_mean[(np.array(r_left) > r_left_cutoff[i])&(z_left>z_limit), :], axis=0), c='green', lw=3)
        ax[i//2, i%2].plot(dF_F_mean[(np.array(r_right) > r_right_cutoff[i])&(z_right>z_limit), :].T, c='red', lw=1, alpha=0.4)
        ax[i//2, i%2].plot(np.mean(dF_F_mean[(np.array(r_right) > r_right_cutoff[i])&(z_right>z_limit), :], axis=0), c='red', lw=3)
    
    # Create custom legend
    legend_elements = [Line2D([0], [0], color='green', lw=3, label=legend_labels[0]),
                       Line2D([0], [0], color='red', lw=3, label=legend_labels[1])]
    fig.legend(handles=legend_elements, loc='upper right')
    plt.show()

def passes_cutoff(row):
    if row['r_max_name'] == 'r0_left':
        return row['r_max'] > r0_left_cutoff
    elif row['r_max_name'] == 'r1_left':
        return row['r_max'] > r1_left_cutoff
    elif row['r_max_name'] == 'r2_left':
        return row['r_max'] > r2_left_cutoff
    elif row['r_max_name'] == 'r3_left':
        return row['r_max'] > r3_left_cutoff
    elif row['r_max_name'] == 'r0_right':
        return row['r_max'] > r0_right_cutoff
    elif row['r_max_name'] == 'r1_right':
        return row['r_max'] > r1_right_cutoff
    elif row['r_max_name'] == 'r2_right':
        return row['r_max'] > r2_right_cutoff
    elif row['r_max_name'] == 'r3_right':
        return row['r_max'] > r3_right_cutoff
    else:
        return False

def determine_function(row):
    if row['passes_cutoff']:
        if 'r0' in row['r_max_name']:
            return 'dynamic_threshold'
        elif 'r1' in row['r_max_name']:
            return 'integrator'
        elif 'r2' in row['r_max_name']:
            return 'motor_command'
        elif 'r3' in row['r_max_name']:
            return 'integrator'
    return 'none'


# Calculate deltaF/F for different conditions
dF_F_rdms_left = calc_dF_F(F_rdms_left)
dF_F_rdms_right = calc_dF_F(F_rdms_right)
dF_F_mean_rdms_right = np.nanmean(dF_F_rdms_right, axis=0)
dF_F_mean_rdms_left = np.nanmean(dF_F_rdms_left, axis=0)
dF_F_no_motion = calc_dF_F(F_no_motion)
dF_F_mean_no_motion = np.nanmean(dF_F_no_motion, axis=0)
dF_F_rdms_left_right = calc_dF_F(F_rdms_left_right)
dF_F_rdms_right_left = calc_dF_F(F_rdms_right_left)
dF_F_mean_rdms_left_right = np.nanmean(dF_F_rdms_left_right, axis=0)
dF_F_mean_rdms_right_left = np.nanmean(dF_F_rdms_right_left, axis=0)
dF_F_ramping_right = calc_dF_F(F_ramping_right)
dF_F_ramping_left = calc_dF_F(F_ramping_left)
dF_F_mean_ramping_right = np.nanmean(dF_F_ramping_right, axis=0)
dF_F_mean_ramping_left = np.nanmean(dF_F_ramping_left, axis=0)

# Function to calculate z-score


# Calculate z-scores for different conditions
rdms_left_zu, rdms_left_zd = z_scorer(dF_F_rdms_left, dF_F_rdms_right, prestim=40, stim=60)  # right z-scored
rdms_right_zu, rdms_right_zd = z_scorer(dF_F_rdms_right, dF_F_rdms_left, prestim=40, stim=60)  # left z-scored



# Cut data to correct length
dF_F_cut_rdms_left = dF_F_rdms_left[:, :, 20:120]
dF_F_cut_rdms_right = dF_F_rdms_right[:, :, 20:120]
dF_F_mean_cut_rdms_left = dF_F_mean_rdms_left[:, 20:100]
dF_F_mean_cut_rdms_right = dF_F_mean_rdms_right[:, 20:100]

dF_F_mean_cut_rdms_all = np.concatenate([dF_F_mean_cut_rdms_left, dF_F_mean_cut_rdms_right], axis=0)
dF_F_mean_cut_rdms_all = savgol_filter(dF_F_mean_cut_rdms_all, 10, 3)

dF_F_mean_rdms_all = np.concatenate([dF_F_mean_rdms_left[:, 20:], dF_F_mean_rdms_right[:, 20:]], axis=0)
dF_F_mean_rdms_all = savgol_filter(dF_F_mean_rdms_all, 10, 3)

dF_F_cut_shift_rdms_left = dF_F_cut_rdms_left - dF_F_cut_rdms_left[:, :, 0][:, :, np.newaxis]
dF_F_cut_shift_rdms_right = dF_F_cut_rdms_left - dF_F_cut_rdms_left[:, :, 0][:, :, np.newaxis]
dF_F_mean_cut_shift_rdms_left = dF_F_mean_cut_rdms_left - dF_F_mean_cut_rdms_left[:, 0][:, np.newaxis]
dF_F_mean_cut_shift_rdms_right = dF_F_mean_cut_rdms_right - dF_F_mean_cut_rdms_right[:, 0][:, np.newaxis]

regressors_cut = regressors[:, :80]
regressors_cut_shift = regressors_cut - regressors_cut[:, 0][:, np.newaxis]

import scipy
r0_all = [float(scipy.stats.pearsonr(regressors_cut[0], dF_F_mean_cut_rdms_all[x, :])[0]) for x in range(dF_F_mean_cut_rdms_all.shape[0])]
r1_all = [float(scipy.stats.pearsonr(regressors_cut[1], dF_F_mean_cut_rdms_all[x, :])[0]) for x in range(dF_F_mean_cut_rdms_all.shape[0])]
r2_all = [float(scipy.stats.pearsonr(regressors_cut[2], dF_F_mean_cut_rdms_all[x, :])[0]) for x in range(dF_F_mean_cut_rdms_all.shape[0])]
r3_all = [float(scipy.stats.pearsonr(regressors_cut[3], dF_F_mean_cut_rdms_all[x, :])[0]) for x in range(dF_F_mean_cut_rdms_all.shape[0])]

r0_left = [float(scipy.stats.pearsonr(regressors_cut[0], dF_F_mean_cut_rdms_left[x, :])[0]) for x in range(dF_F_mean_cut_rdms_left.shape[0])]
r1_left = [float(scipy.stats.pearsonr(regressors_cut[1], dF_F_mean_cut_rdms_left[x, :])[0]) for x in range(dF_F_mean_cut_rdms_left.shape[0])]
r2_left = [float(scipy.stats.pearsonr(regressors_cut[2], dF_F_mean_cut_rdms_left[x, :])[0]) for x in range(dF_F_mean_cut_rdms_left.shape[0])]
r3_left = [float(scipy.stats.pearsonr(regressors_cut[3], dF_F_mean_cut_rdms_left[x, :])[0]) for x in range(dF_F_mean_cut_rdms_left.shape[0])]

r0_right = [float(scipy.stats.pearsonr(regressors_cut[0], dF_F_mean_cut_rdms_right[x, :])[0]) for x in range(dF_F_mean_cut_rdms_right.shape[0])]
r1_right = [float(scipy.stats.pearsonr(regressors_cut[1], dF_F_mean_cut_rdms_right[x, :])[0]) for x in range(dF_F_mean_cut_rdms_right.shape[0])]
r2_right = [float(scipy.stats.pearsonr(regressors_cut[2], dF_F_mean_cut_rdms_right[x, :])[0]) for x in range(dF_F_mean_cut_rdms_right.shape[0])]
r3_right = [float(scipy.stats.pearsonr(regressors_cut[3], dF_F_mean_cut_rdms_right[x, :])[0]) for x in range(dF_F_mean_cut_rdms_right.shape[0])]


#reconcile several regressors so a cell is not classified as two different types




r0_all_cutoff = np.percentile(r0_all, 99)
r1_all_cutoff = np.percentile(r1_all, 99)
r2_all_cutoff = np.percentile(r2_all, 99)
r3_all_cutoff = np.percentile(r3_all, 99)
r0_left_cutoff = np.percentile(r0_left, 99)
r1_left_cutoff = np.percentile(r1_left, 99)
r2_left_cutoff = np.percentile(r2_left, 99)
r3_left_cutoff = np.percentile(r3_left, 99)
r0_right_cutoff = np.percentile(r0_right, 99)
r1_right_cutoff = np.percentile(r1_right, 99)
r2_right_cutoff = np.percentile(r2_right, 99)
r3_right_cutoff = np.percentile(r3_right, 99)

r0_all, r1_all, r2_all, r3_all = reconcile_regressors(r0_all, r1_all, r2_all, r3_all, threshold=0.02)
r0_left, r1_left, r2_left, r3_left = reconcile_regressors(r0_left, r1_left, r2_left, r3_left, threshold=0.02)
r0_right, r1_right, r2_right, r3_right = reconcile_regressors(r0_right, r1_right, r2_right, r3_right, threshold=0.02)


all_fish_df['r0_left'] = r0_left
all_fish_df['r1_left'] = r1_left
all_fish_df['r2_left'] = r2_left
all_fish_df['r3_left'] = r3_left
all_fish_df['r0_right'] = r0_right
all_fish_df['r1_right'] = r1_right
all_fish_df['r2_right'] = r2_right
all_fish_df['r3_right'] = r3_right

# Find the maximum r value and its corresponding name for each row
all_fish_df['r_max'] = all_fish_df[['r0_left', 'r1_left', 'r2_left', 'r3_left', 'r0_right', 'r1_right', 'r2_right', 'r3_right']].max(axis=1)
all_fish_df['r_max_name'] = all_fish_df[['r0_left', 'r1_left', 'r2_left', 'r3_left', 'r0_right', 'r1_right', 'r2_right', 'r3_right']].idxmax(axis=1)
# Determine if r_max passes the corresponding cutoff and write True or False into a new column


all_fish_df['passes_cutoff'] = all_fish_df.apply(passes_cutoff, axis=1)


all_fish_df['functional_type'] = all_fish_df.apply(determine_function, axis=1)
color_dict = {'dynamic_threshold': '#68C7EC', 'integrator': '#ED7658', 'motor_command': '#7F58AF', 'none': 'gray'}

plot_cells_in_brain(all_fish_df, color_dict, cmap='gray',seg_type='unit_contours')
plt.show()
# Plotting the results

fig,ax = plt.subplots(2,2,figsize=(10,10))
ax[0,0].plot(dF_F_mean_rdms_all[np.array(r0_all)>r0_all_cutoff,:].T,c='gray',lw=1,alpha=0.5)
ax[0,0].plot(regressors_cut_shift[0],lw=5,alpha=0.6,c='red')
ax[0,1].plot(dF_F_mean_rdms_all[np.array(r1_all)>r1_all_cutoff,:].T,c='gray',lw=1,alpha=0.5)
ax[0,1].plot(regressors_cut_shift[1],lw=5,alpha=0.6,c='red')
ax[1,0].plot(dF_F_mean_rdms_all[np.array(r2_all)>r2_all_cutoff,:].T,c='gray',lw=1,alpha=0.5)
ax[1,0].plot(regressors_cut_shift[2],lw=5,alpha=0.6,c='red')
ax[1,1].plot(dF_F_mean_rdms_all[np.array(r3_all)>r3_all_cutoff,:].T,c='gray',lw=1,alpha=0.5)
ax[1,1].plot(regressors_cut_shift[3],lw=5,alpha=0.6,c='red')
plt.show()

fig,ax = plt.subplots(2,2,figsize=(10,10))
fig.suptitle('Mean dF/F Responses and Regressors RDMS left/right')
ax[0,0].plot(np.mean(dF_F_mean_rdms_all[np.array(r0_all)>r0_all_cutoff,:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[0,0].plot(regressors_cut_shift[0],lw=5,alpha=0.6,c='red')
ax[0,1].plot(np.mean(dF_F_mean_rdms_all[np.array(r1_all)>r1_all_cutoff,:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[0,1].plot(regressors_cut_shift[1],lw=5,alpha=0.6,c='red')
ax[1,0].plot(np.mean(dF_F_mean_rdms_all[np.array(r2_all)>r2_all_cutoff,:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[1,0].plot(regressors_cut_shift[2],lw=5,alpha=0.6,c='red')
ax[1,1].plot(np.mean(dF_F_mean_rdms_all[np.array(r3_all)>r3_all_cutoff,:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[1,1].plot(regressors_cut_shift[3],lw=5,alpha=0.6,c='red')
plt.show()



plot_dF_F_responses(dF_F_mean_no_motion, 
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right],
                    [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                    [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
                    rdms_left_zu,rdms_right_zu,
                    'Mean dF/F Responses RDMS no motion', 
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])
plot_dF_F_responses(dF_F_mean_rdms_left_right, 
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right],
                    [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                    [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff], 
                    rdms_left_zu,rdms_right_zu,
                    'dF/F Responses RDMS left > RDMS right', 
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])
plot_dF_F_responses(dF_F_mean_rdms_right_left, 
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right],
                    [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                    [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
                    rdms_left_zu,rdms_right_zu,
                    'dF/F Responses RDMS right > RDMS left', 
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])
plot_dF_F_responses(dF_F_mean_ramping_right, 
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right],
                    [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                    [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff], 
                    rdms_left_zu,rdms_right_zu,
                    'dF/F Responses ramping right', 
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])
plot_dF_F_responses(dF_F_mean_ramping_left, 
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right],
                    [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                    [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff], 
                    rdms_left_zu,rdms_right_zu,
                    'dF/F Responses ramping left', 
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])



#write dynamics to hdf5 file
first_level = ['fish0', 'fish1','all']
second_level = ['dF_F_rdms_left', 'dF_F_rdms_right', 'dF_F_mean_rdms_right', 'dF_F_mean_rdms_left', 'dF_F_no_motion', 'dF_F_mean_no_motion', 'dF_F_rdms_left_right', 'dF_F_rdms_right_left', 'dF_F_mean_rdms_left_right', 'dF_F_mean_rdms_right_left', 'dF_F_ramping_right', 'dF_F_ramping_left', 'dF_F_mean_ramping_right', 'dF_F_mean_ramping_left']
third_level = ['all','all_passing','dynamic_threshold','integrator','motor_command','none']
with h5py.File(path_fish0.parent/'dF_F_dynamics.hdf5', "w") as f:

    for lvl1 in first_level:
        for lvl2 in second_level:
            for lvl3 in third_level:
                temp = np.full(all_fish_df.shape[0],fill_value=True)
                if lvl1 != 'all':
                    temp = (all_fish_df['fish_id']== lvl1) * temp
                
                if lvl3 in ['dynamic_threshold','integrator','motor_command','none']:
                    temp = (all_fish_df['functional_type']==lvl3) * temp
                elif lvl3 == 'all_passing':
                    temp = (all_fish_df['passes_cutoff']==True) * temp

                if len(eval(lvl2).shape) == 3:
                    data = eval(lvl2)[:,temp,:]
                elif len(eval(lvl2).shape) == 2:
                    data = eval(lvl2)[temp,:]

                f.create_dataset(f'{lvl1}/{lvl2}/{lvl3}', data=data, dtype=data.dtype)

all_fish_df.loc[:,['unit_name','fish_id','functional_type']].to_hdf(path_fish0.parent/'dF_F_dynamics.hdf5', key='df_correlation_classification', mode='a')