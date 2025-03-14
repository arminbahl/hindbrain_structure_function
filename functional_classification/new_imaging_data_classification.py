import h5py
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path
from numba import jit
import numpy as np
import pylab as pl
import pathlib

# matplotlib.use("macosx")
import matplotlib.colors as colors
import numpy as np
import scipy

from analysis_helpers.analysis.utils.figure_helper import Figure



#Set path pointing at folder containing the data
path = Path(r'Z:\Kim\clem_g8s\dot_motion_coherence_direction_change')  # Z: = imaging_data1/M11 2P microscopes
path_fish0 = path / r"2025-03-13_14-43-43_fish000_setup0_arena0_plane1\2025-03-13_14-43-43_fish000_setup0_arena0_plane1_preprocessed_data.h5"
path_fish1 = path / r"2025-03-13_12-56-57_fish001_setup1_arena0\2025-03-13_12-56-57_fish001_setup1_arena0_preprocessed_data.h5"
path_fish3 = path / r"2025-03-13_16-49-45_fish003_setup0_arena0\2025-03-13_16-49-45_fish003_setup0_arena0_preprocessed_data.h5"
path_fish4 = path / r"2025-03-13_19-15-32_fish004_setup1_arena0\2025-03-13_19-15-32_fish004_setup1_arena0_preprocessed_data.h5"
# Load regressors
regressors = np.load(r"C:\Users\ag-bahl\Desktop\kmeans_regressors.npy")*100

all_fish_df = pd.DataFrame()
F_fish_rdms_left, F_fish_rdms_right, F_fish_rdms_right, F_fish_rdms_left_right, F_fish_rdms_right_left = [],[],[],[],[]
# Load data from HDF5 files
for idx, path_fish in {0:path_fish0, 1:path_fish1, 3:path_fish3, 4:path_fish4}.items():
    with h5py.File(path_fish) as f:
        keys = [key for key in f.keys() if key.startswith("repeat")]
        print('Fish', idx, 'keys', keys)
        for key in keys:
            F_fish_rdms_left.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/left_left/F'])[:12,...])
            F_fish_rdms_right.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/right_right/F'])[:12,...])
            F_fish_rdms_left_right.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/left_right/F'])[:12,...])
            F_fish_rdms_right_left.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/right_left/F'])[:12,...])

            average_image_fish = np.array(f[f'{key}/preprocessed_data/fish00/imaging_data_channel0_time_averaged'])
            unit_names_fish = np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/unit_names'])
            hdf5_path_fish = np.full(unit_names_fish.shape, fill_value=str(path_fish))
            fish_id_fish = np.full(unit_names_fish.shape, fill_value=f'fish{idx}')
            plane_fish = np.full(unit_names_fish.shape, fill_value=f'{key.split('_')[2]}')
            fish_df = pd.DataFrame([unit_names_fish, hdf5_path_fish, fish_id_fish, plane_fish]).T
            fish_df.columns = ['unit_name', 'hdf5_path', 'fish_id', 'plane']
            all_fish_df = pd.concat([all_fish_df, fish_df], axis=0)

# Combine data from both fish
F_rdms_left = np.concatenate(F_fish_rdms_left, axis=1)[:,:,:]
F_rdms_right = np.concatenate(F_fish_rdms_right, axis=1)[:,:,:]
F_rdms_left_right = np.concatenate(F_fish_rdms_left_right, axis=1)[:,:,:]
F_rdms_right_left = np.concatenate(F_fish_rdms_right_left, axis=1)[:,:,:]

def calc_dF_F(F, dt=0.5):
    """
    Calculate deltaF/F for given fluorescence data.
    
    Parameters:
    F (numpy array): Fluorescence data
    dt (float): Time step
    
    Returns:
    numpy array: deltaF/F data
    """
    F0 = np.nanmean(F[:, :, 40:80], axis=2, keepdims=True)
    dF_F = 100 * (F - F0) / F0
    return dF_F

def retrieve_segmentation(path_hdf5,id,plane='z000',seg_type='unit_contours'):
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
        seg = np.array(f[f'repeat00_tile000_{plane}_950nm/preprocessed_data/fish00/cellpose_segmentation/{seg_type}/{id}'])
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

    for (fish, plane) in df[['fish_id', 'plane']].drop_duplicates().itertuples(index=False):
        passing_cells_per_fish_plane = df[(df['fish_id']==fish)&(df['plane']==plane)&(df['passes_cutoff']==True)]

        fig,ax = plt.subplots(1,1,figsize=(10,10))
        with h5py.File(passing_cells_per_fish_plane['hdf5_path'].unique()[0]) as f:
            average_image = np.array(f[f'repeat00_tile000_{plane}_950nm/preprocessed_data/fish00/imaging_data_channel0_time_averaged'])
            average_image = np.clip(average_image,np.percentile(average_image,2),np.percentile(average_image,98))
        ax.imshow(average_image,cmap=cmap)

        for i,item in passing_cells_per_fish_plane.iterrows():
            cell_seg = retrieve_segmentation(item['hdf5_path'],item['unit_name'],plane,seg_type)
            
            ax.plot(cell_seg[:,0],cell_seg[:,1],color=color_dict[item['functional_type']],lw=1)
        fig.show()

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

        if i == 0:
            left_responses_MON = dF_F_mean[(np.array(r_left) > r_left_cutoff[i]) & (z_left > z_limit), :].T.copy()
            right_responses_MON = dF_F_mean[(np.array(r_right) > r_right_cutoff[i])&(z_right>z_limit), :].T.copy()
        elif i == 1:
            left_responses_MI1 = dF_F_mean[(np.array(r_left) > r_left_cutoff[i]) & (z_left > z_limit), :].T.copy()
            right_responses_MI1 = dF_F_mean[(np.array(r_right) > r_right_cutoff[i]) & (z_right > z_limit), :].T.copy()
        elif i == 2:
            left_responses_SMI = dF_F_mean[(np.array(r_left) > r_left_cutoff[i]) & (z_left > z_limit), :].T.copy()
            right_responses_SMI = dF_F_mean[(np.array(r_right) > r_right_cutoff[i]) & (z_right > z_limit), :].T.copy()
        elif i == 3:
            left_responses_MI2 = dF_F_mean[(np.array(r_left) > r_left_cutoff[i]) & (z_left > z_limit), :].T.copy()
            right_responses_MI2 = dF_F_mean[(np.array(r_right) > r_right_cutoff[i]) & (z_right > z_limit), :].T.copy()

        ax[i//2, i%2].plot(dF_F_mean[(np.array(r_left) > r_left_cutoff[i])&(z_left>z_limit), :].T, c='green', lw=1, alpha=0.4)
        ax[i//2, i%2].plot(np.mean(dF_F_mean[(np.array(r_left) > r_left_cutoff[i])&(z_left>z_limit), :], axis=0), c='green', lw=3)
        ax[i//2, i%2].plot(dF_F_mean[(np.array(r_right) > r_right_cutoff[i])&(z_right>z_limit), :].T, c='red', lw=1, alpha=0.4)
        ax[i//2, i%2].plot(np.mean(dF_F_mean[(np.array(r_right) > r_right_cutoff[i])&(z_right>z_limit), :], axis=0), c='red', lw=3)
    
    # Create custom legend
    legend_elements = [Line2D([0], [0], color='green', lw=3, label=legend_labels[0]),
                       Line2D([0], [0], color='red', lw=3, label=legend_labels[1])]
    fig.legend(handles=legend_elements, loc='upper right')
    plt.show()

    return (left_responses_MON,
            right_responses_MON,
            np.c_[left_responses_MI1, left_responses_MI2],
            np.c_[right_responses_MI1, right_responses_MI2],
            left_responses_SMI,
            right_responses_SMI)

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
# dF_F_no_motion = calc_dF_F(F_no_motion)
# dF_F_mean_no_motion = np.nanmean(dF_F_no_motion, axis=0)
dF_F_rdms_left_right = calc_dF_F(F_rdms_left_right)
dF_F_rdms_right_left = calc_dF_F(F_rdms_right_left)
dF_F_mean_rdms_left_right = np.nanmean(dF_F_rdms_left_right, axis=0)
dF_F_mean_rdms_right_left = np.nanmean(dF_F_rdms_right_left, axis=0)
# dF_F_ramping_right = calc_dF_F(F_ramping_right)
# dF_F_ramping_left = calc_dF_F(F_ramping_left)
# dF_F_mean_ramping_right = np.nanmean(dF_F_ramping_right, axis=0)
# dF_F_mean_ramping_left = np.nanmean(dF_F_ramping_left, axis=0)

# Function to calculate z-score

# Calculate z-scores for different conditions
rdms_left_zu, rdms_left_zd = z_scorer(dF_F_rdms_left, dF_F_rdms_right, prestim=80, stim=60)  # right z-scored
rdms_right_zu, rdms_right_zd = z_scorer(dF_F_rdms_right, dF_F_rdms_left, prestim=80, stim=60)  # left z-scored



# Cut data to correct length
# dF_F_cut_rdms_left = dF_F_rdms_left[:, :, 20:120]
# dF_F_cut_rdms_right = dF_F_rdms_right[:, :, 20:120]
# dF_F_mean_cut_rdms_left = dF_F_mean_rdms_left[:, 20:100]
# dF_F_mean_cut_rdms_right = dF_F_mean_rdms_right[:, 20:100]

dF_F_cut_rdms_left = dF_F_rdms_left[:, :, :]
dF_F_cut_rdms_right = dF_F_rdms_right[:, :, :]

# Mean over trial, cut to 80 datapoints, to match the length of the regressor (30 s of stimulation)
dF_F_mean_cut_rdms_left = dF_F_mean_rdms_left[:, 60:140]
dF_F_mean_cut_rdms_right = dF_F_mean_rdms_right[:, 60:140]

dF_F_mean_cut_rdms_all = np.concatenate([dF_F_mean_cut_rdms_left, dF_F_mean_cut_rdms_right], axis=0)
dF_F_mean_cut_rdms_all = savgol_filter(dF_F_mean_cut_rdms_all, 10, 3)

dF_F_mean_rdms_all = np.concatenate([dF_F_mean_rdms_left[:, :], dF_F_mean_rdms_right[:, :]], axis=0)
dF_F_mean_rdms_all = savgol_filter(dF_F_mean_rdms_all, 10, 3)

dF_F_cut_shift_rdms_left = dF_F_cut_rdms_left - dF_F_cut_rdms_left[:, :, 0][:, :, np.newaxis]
dF_F_cut_shift_rdms_right = dF_F_cut_rdms_left - dF_F_cut_rdms_left[:, :, 0][:, :, np.newaxis]
dF_F_mean_cut_shift_rdms_left = dF_F_mean_cut_rdms_left - dF_F_mean_cut_rdms_left[:, 0][:, np.newaxis]
dF_F_mean_cut_shift_rdms_right = dF_F_mean_cut_rdms_right - dF_F_mean_cut_rdms_right[:, 0][:, np.newaxis]

# Make the regressors 40 s long (10 s base line + 30 s)
regressors_cut = regressors[:, :80]
regressors_cut_shift = regressors_cut - regressors_cut[:, 0][:, np.newaxis]
# import pylab as pl
# pl.plot(regressors_cut[0])
# pl.plot(regressors_cut[1])
# pl.plot(regressors_cut[2])
# pl.plot(regressors_cut[3])
# pl.show()
# sdf
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

########################################
# Plot and analyze

# plot_dF_F_responses(dF_F_mean_no_motion,
#                     [r0_left, r1_left, r2_left, r3_left],
#                     [r0_right, r1_right, r2_right, r3_right],
#                     [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
#                     [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
#                     rdms_left_zu,rdms_right_zu,
#                     'Mean dF/F Responses RDMS no motion',
#                     ['Cells responding to RDMS left', 'Cells responding to RDMS right'])

(left_responses_MON_left_left,
right_responses_MON_left_left,
left_responses_MI_left_left,
right_responses_MI_left_left,
left_responses_SMI_left_left,
right_responses_SMI_left_left) =  plot_dF_F_responses(dF_F_mean_rdms_left,
                    [r0_left, r1_left, r2_left, r3_left],
                    [r0_right, r1_right, r2_right, r3_right],
                    [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                    [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
                    rdms_left_zu,rdms_right_zu,
                    'Mean dF/F Responses RDMS left, left',
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])

(left_responses_MON_right_right,
 right_responses_MON_right_right,
 left_responses_MI_right_right,
 right_responses_MI_right_right,
 left_responses_SMI_right_right,
 right_responses_SMI_right_right) = plot_dF_F_responses(dF_F_mean_rdms_right,
                    [r0_left, r1_left, r2_left, r3_left],
                    [r0_right, r1_right, r2_right, r3_right],
                    [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                    [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
                    rdms_left_zu,rdms_right_zu,
                    'Mean dF/F Responses RDMS right, right',
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])

(left_responses_MON_left_right,
 right_responses_MON_left_right,
 left_responses_MI_left_right,
 right_responses_MI_left_right,
 left_responses_SMI_left_right,
 right_responses_SMI_left_right) = plot_dF_F_responses(dF_F_mean_rdms_left_right,
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right],
                    [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                    [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff], 
                    rdms_left_zu,rdms_right_zu,
                    'dF/F Responses RDMS left, right',
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])
(left_responses_MON_right_left,
 right_responses_MON_right_left,
 left_responses_MI_right_left,
 right_responses_MI_right_left,
 left_responses_SMI_right_left,
 right_responses_SMI_right_left) = plot_dF_F_responses(dF_F_mean_rdms_right_left,
                    [r0_left, r1_left, r2_left, r3_left], 
                    [r0_right, r1_right, r2_right, r3_right],
                    [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                    [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
                    rdms_left_zu,rdms_right_zu,
                    'dF/F Responses RDMS right, left',
                    ['Cells responding to RDMS left', 'Cells responding to RDMS right'])

# plot_dF_F_responses(dF_F_mean_ramping_right,
#                     [r0_left, r1_left, r2_left, r3_left],
#                     [r0_right, r1_right, r2_right, r3_right],
#                     [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
#                     [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
#                     rdms_left_zu,rdms_right_zu,
#                     'dF/F Responses ramping right',
#                     ['Cells responding to RDMS left', 'Cells responding to RDMS right'])
#
# plot_dF_F_responses(dF_F_mean_ramping_left,
#                     [r0_left, r1_left, r2_left, r3_left],
#                     [r0_right, r1_right, r2_right, r3_right],
#                     [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
#                     [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
#                     rdms_left_zu,rdms_right_zu,
#                     'dF/F Responses ramping left',
#                     ['Cells responding to RDMS left', 'Cells responding to RDMS right'])


MON_left_left = np.c_[left_responses_MON_left_left, right_responses_MON_right_right]
MI_left_left = np.c_[left_responses_MI_left_left, right_responses_MI_right_right]
SMI_left_left = np.c_[left_responses_SMI_left_left, right_responses_SMI_right_right]

MON_left_right = np.c_[left_responses_MON_left_right, right_responses_MON_right_left]
MI_left_right = np.c_[left_responses_MI_left_right, right_responses_MI_right_left]
SMI_left_right = np.c_[left_responses_SMI_left_right, right_responses_SMI_right_left]


## Plotting and stats
# Do a time constant fit (average time to 90% as in

dt = 0.5
t = np.arange(0, 110, dt)
S_left1 = np.zeros(len(t))
S_right1 = np.zeros(len(t))
S_left2 = np.zeros(len(t))
S_right2 = np.zeros(len(t))

# Both visual inputs have some baseline input rate
S_left1[:] = 0.5
S_right1[:] = 0.5

# Show some input on the left eye
S_left1[int(40/dt):int(70/dt)] += 1
# Which reduced firing on the right side.
S_right1[int(40/dt):int(70/dt)] -= 0.5

S_left2[:] = 0.5
S_right2[:] = 0.5

# Show some input on the left eye
# S_left2[int(20/dt):int(40/dt)] += np.linspace(0, 1, int(20/dt))
# S_right2[int(20/dt):int(40/dt)] -= 0.5*S_left2[int(20/dt):int(40/dt)]

# Opposing stimulation
S_left2[int(40/dt):int(55/dt)] += 1
S_right2[int(40/dt):int(55/dt)] -= 0.5
S_left2[int(55/dt):int(70/dt)] -= 0.5
S_right2[int(55/dt):int(70/dt)] += 1

# Make a standard figure
fig = Figure(figure_title="Figure 5_data")

colors = ["#68C7EC", "#ED7658", "#7F58AF"]
line_dashes = [None, None, None, None]

names = ["MON", "MI", "SMI"]

for prediction in [0, 1]:

    plot0 = fig.create_plot(plot_label='a', xpos=3.5 + prediction * 8, ypos=24.5, plot_height=0.5, plot_width=2,
                            errorbar_area=True,
                            xl="", xmin=9, xmax=111, xticks=[],
                            plot_title="Ipsi. hemisphere",
                            yl="", ymin=-0.1, ymax=1.6, yticks=[0, 0.5], hlines=[0])

    plot1 = fig.create_plot(plot_label='a', xpos=3.5  + prediction * 8, ypos=22, plot_height=2, plot_width=2, errorbar_area=True,
                            xl="Time (s)", xmin=9, xmax=111, xticks=[10, 20], vlines=[40,55,70],
                            yl="ΔF / F0", ymin=-21, ymax=71.1, yticks=[0, 35, 71], hlines=[0])

    if prediction == 0:
        plot0.draw_line(x=t[int(10 / dt):], y=S_left1[int(10 / dt):], lw=1.5, lc='black')
    if prediction == 1:
        plot0.draw_line(x=t[int(10 / dt):], y=S_left2[int(10 / dt):], lw=1.5, lc='black')

    delays = []
    delays2 = []

    for i_cell in np.arange(3):
        if i_cell == 0 and prediction == 0:
            df_f0 = MON_left_left.mean(axis=1)
        if i_cell == 0 and prediction == 1:
            df_f0 = MON_left_right.mean(axis=1)

        if i_cell == 1 and prediction == 0:
            df_f0 = MI_left_left.mean(axis=1)
        if i_cell == 1 and prediction == 1:
            df_f0 = MI_left_right.mean(axis=1)

        if i_cell == 2 and prediction == 0:
            df_f0 = SMI_left_left.mean(axis=1)
        if i_cell == 2 and prediction == 1:
            df_f0 = SMI_left_right.mean(axis=1)

        if prediction == 0:
            rate_90_percent = df_f0[int(40 / dt):int(70 / dt)].max() * 0.9
            ind = np.where(df_f0[int(40 / dt):int(70 / dt)] > rate_90_percent)
        else:
            rate_90_percent = df_f0[int(40 / dt):int(55 / dt)].max() * 0.9
            ind = np.where(df_f0[int(40 / dt):int(55 / dt)] > rate_90_percent)

        if len(ind[0]) > 0:
            t_to_90_percent = ind[0][0] * dt
            delays.append(t_to_90_percent)
        else:
            delays.append(np.nan)

        if prediction == 0:
            val_10_percent = df_f0[int(70 / dt)] * 0.1
            ind = np.where(df_f0[int(70 / dt):] < val_10_percent)
        else:
            val_10_percent = df_f0[int(55/dt)]*0.1
            ind = np.where(df_f0[int(55/dt):] < val_10_percent)

        if len(ind[0]) > 0:
            t_to_10_percent = ind[0][0] * dt
            delays2.append(t_to_10_percent)
        else:
            delays2.append(np.nan)

        plot1.draw_line(x=t[int(10/dt):], y=df_f0[int(10/dt):], lw=1.5, lc=colors[i_cell], line_dashes=line_dashes[i_cell], label=f"{names[i_cell]}")

    plot2 = fig.create_plot(plot_label='a', xpos=3.5 + prediction * 8, ypos=13.5, plot_height=1.5,
                            plot_width=1.5,
                            errorbar_area=True,
                            xl="Time to 90% of max (s)", xmin=-0.1, xmax=35.1, xticks=[0, 10, 20],
                            yl="Time from max to 10% (s)", ymin=-0.1, ymax=35.1, yticks=[0, 10, 20], hlines=[0], vlines=[0])
    plot2.draw_line([0, 20], [0,20], lw=0.5, line_dashes=(2,2), lc='gray')
    print(delays, delays2)

    for i in range(3):
        plot2.draw_scatter(x=[delays[i]], y=[delays2[i]], pc=colors[i])

fig.save(Path.home() / 'Desktop' / "fig_test_data.pdf", open_file=True)
