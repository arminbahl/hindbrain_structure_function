
import h5py
import pandas as pd
import numpy as np
import sys
import os
sys.path.extend(['/Users/fkampf/PycharmProjects'])
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load regressors
regressors = np.load('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud/make_figures_FK_output/functional_analysis/kmeans_regressors.npy')*100

# Load data from HDF5 files
with h5py.File('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud/clem_zfish1/activity_recordings/untitled folder/2025-03-05_13-12-33_fish000_setup0_arena0_AB_preprocessed_data.h5') as f:
    F_fish0_rdms_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/left_left/F'])
    F_fish0_rdms_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/right_right/F'])
    F_fish0_no_motion = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/no_motion/F'])
    F_fish0_ramping_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/ramping_right/F'])
    F_fish0_ramping_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/ramping_left/F'])
    F_fish0_rdms_left_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/left_right/F'])
    F_fish0_rdms_right_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/right_left/F'])

with h5py.File('/Users/fkampf/Documents/hindbrain_structure_function/nextcloud/clem_zfish1/activity_recordings/untitled folder/2025-03-05_13-12-40_fish001_setup1_arena0_AB_preprocessed_data.h5') as f:
    F_fish1_rdms_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/left_left/F'])
    F_fish1_rdms_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/right_right/F'])
    F_fish1_rdms_no_motion = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/no_motion/F'])
    F_fish1_ramping_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/ramping_right/F'])
    F_fish1_ramping_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/ramping_left/F'])
    F_fish1_rdms_left_right = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/left_right/F'])
    F_fish1_rdms_right_left = np.array(f['repeat00_tile000_z000_950nm/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/right_left/F'])

# Combine data from both fish
F_rdms_left = np.concatenate([F_fish0_rdms_left,F_fish1_rdms_left], axis=1)[:,:,:120]
F_rdms_right = np.concatenate([F_fish0_rdms_right,F_fish1_rdms_right[:10,:,:]], axis=1)[:,:,:120]
F_no_motion = np.concatenate([F_fish0_no_motion, F_fish1_rdms_no_motion], axis=1)[:,:,:120]
F_ramping_right = np.concatenate([F_fish0_ramping_right, F_fish1_ramping_right], axis=1)[:,:,:120]
F_ramping_left = np.concatenate([F_fish0_ramping_left, F_fish1_ramping_left], axis=1)[:,:,:120]
F_rdms_left_right = np.concatenate([F_fish0_rdms_left_right, F_fish1_rdms_left_right], axis=1)[:,:,:120]
F_rdms_right_left = np.concatenate([F_fish0_rdms_right_left[:-1], F_fish1_rdms_right_left], axis=1)[:,:,:120]

# Function to calculate deltaF/F
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



# Plotting the results

fig,ax = plt.subplots(2,2,figsize=(10,10))
ax[0,0].plot(dF_F_mean_rdms_all[np.array(r0_all)>np.percentile(r0_all,99),:].T,c='gray',lw=1,alpha=0.5)
ax[0,0].plot(regressors_cut_shift[0],lw=5,alpha=0.6,c='red')
ax[0,1].plot(dF_F_mean_rdms_all[np.array(r1_all)>np.percentile(r1_all,99),:].T,c='gray',lw=1,alpha=0.5)
ax[0,1].plot(regressors_cut_shift[1],lw=5,alpha=0.6,c='red')
ax[1,0].plot(dF_F_mean_rdms_all[np.array(r2_all)>np.percentile(r2_all,99),:].T,c='gray',lw=1,alpha=0.5)
ax[1,0].plot(regressors_cut_shift[2],lw=5,alpha=0.6,c='red')
ax[1,1].plot(dF_F_mean_rdms_all[np.array(r3_all)>np.percentile(r3_all,99),:].T,c='gray',lw=1,alpha=0.5)
ax[1,1].plot(regressors_cut_shift[3],lw=5,alpha=0.6,c='red')
plt.show()

fig,ax = plt.subplots(2,2,figsize=(10,10))
fig.suptitle('Mean dF/F Responses and Regressors RDMS left/right')
ax[0,0].plot(np.mean(dF_F_mean_rdms_all[np.array(r0_all)>np.percentile(r0_all,99),:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[0,0].plot(regressors_cut_shift[0],lw=5,alpha=0.6,c='red')
ax[0,1].plot(np.mean(dF_F_mean_rdms_all[np.array(r1_all)>np.percentile(r1_all,99),:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[0,1].plot(regressors_cut_shift[1],lw=5,alpha=0.6,c='red')
ax[1,0].plot(np.mean(dF_F_mean_rdms_all[np.array(r2_all)>np.percentile(r2_all,99),:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[1,0].plot(regressors_cut_shift[2],lw=5,alpha=0.6,c='red')
ax[1,1].plot(np.mean(dF_F_mean_rdms_all[np.array(r3_all)>np.percentile(r3_all,99),:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[1,1].plot(regressors_cut_shift[3],lw=5,alpha=0.6,c='red')
plt.show()


from matplotlib.lines import Line2D

def plot_dF_F_responses(dF_F_mean, r_left_list, r_right_list,z_left,z_right, title, legend_labels,z_limit=3.3):
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
        ax[i//2, i%2].plot(dF_F_mean[(np.array(r_left) > np.percentile(r_left, 99))&(z_left>z_limit), :].T, c='green', lw=1, alpha=0.4)
        ax[i//2, i%2].plot(np.mean(dF_F_mean[(np.array(r_left) > np.percentile(r_left, 99))&(z_left>z_limit), :], axis=0), c='green', lw=3)
        ax[i//2, i%2].plot(dF_F_mean[(np.array(r_right) > np.percentile(r_right, 99))&(z_right>z_limit), :].T, c='red', lw=1, alpha=0.4)
        ax[i//2, i%2].plot(np.mean(dF_F_mean[(np.array(r_right) > np.percentile(r_right, 99))&(z_right>z_limit), :], axis=0), c='red', lw=3)
    
    # Create custom legend
    legend_elements = [Line2D([0], [0], color='green', lw=3, label=legend_labels[0]),
                       Line2D([0], [0], color='red', lw=3, label=legend_labels[1])]
    fig.legend(handles=legend_elements, loc='upper right')
    plt.show()

# Example usage:
plot_dF_F_responses(dF_F_mean_no_motion, 
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right],
                    rdms_left_zu,rdms_right_zu,
                    'Mean dF/F Responses RDMS no motion', 
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])
plot_dF_F_responses(dF_F_mean_rdms_left_right, 
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right], 
                    rdms_left_zu,rdms_right_zu,
                    'dF/F Responses RDMS left > RDMS right', 
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])
plot_dF_F_responses(dF_F_mean_rdms_right_left, 
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right],
                    rdms_left_zu,rdms_right_zu,
                    'dF/F Responses RDMS right > RDMS left', 
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])
plot_dF_F_responses(dF_F_mean_ramping_right, 
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right], 
                    rdms_left_zu,rdms_right_zu,
                    'dF/F Responses ramping right', 
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])
plot_dF_F_responses(dF_F_mean_ramping_left, 
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right], 
                    rdms_left_zu,rdms_right_zu,
                    'dF/F Responses ramping left', 
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])