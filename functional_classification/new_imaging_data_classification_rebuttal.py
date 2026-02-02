import h5py
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path
import os
import re
import pathlib
import scipy

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))

# matplotlib.use("macosx")
import matplotlib.colors as colors

from visualization.figure_helper import Figure



#Set path pointing at folder containing the data
path = Path(r'Z:\Kim\clem_rebuttal')
# Find all HDF5 files in the directory
hdf5_files = [f for f in os.listdir(path) if f.endswith('.h5')]

# Load regressors
regressors = np.load(r"Z:\Kim\clem_rebuttal\kmeans_regressors.npy")*100

all_fish_df = pd.DataFrame()
F_fish_rdms_left_100, F_fish_rdms_left_50, F_fish_rdms_right_100, F_fish_rdms_right_50, F_fish_rdms_oscillating_left, F_fish_rdms_oscillating_right, F_fish_rdms_switching_left, F_fish_rdms_switching_right = [],[],[],[],[],[],[],[]

# Track global cell indices across concatenated arrays
_global_cell_offset = 0

# Load data from HDF5 files
for idx, file in enumerate(hdf5_files):
    path_fish = path / file
    # Extract fish number from file name
    fish_str = re.search(r'fish(\d+)', file).group(1)
    fish_id = int(fish_str)
    with h5py.File(path_fish) as f:
        keys = [key for key in f.keys() if key.startswith("repeat")]
        print(f'Fish {fish_id} keys', keys)
        for key in keys:
            F_fish_rdms_left_100.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/constant_left_100/F']))
            F_fish_rdms_right_100.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/constant_right_100/F']))
            F_fish_rdms_left_50.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/constant_left_50/F']))
            F_fish_rdms_right_50.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/constant_right_50/F']))
            F_fish_rdms_oscillating_left.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/oscillating_left/F']))
            F_fish_rdms_oscillating_right.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/oscillating_right/F']))
            F_fish_rdms_switching_left.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/switching_left/F']))
            F_fish_rdms_switching_right.append(np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/stimulus_aligned_dynamics/switching_right/F']))


            average_image_fish = np.array(f[f'{key}/preprocessed_data/fish00/imaging_data_channel0_time_averaged'])
            unit_names_fish = np.array(f[f'{key}/preprocessed_data/fish00/cellpose_segmentation/unit_names'])
            hdf5_path_fish = np.full(unit_names_fish.shape, fill_value=str(path_fish))
            fish_id_fish = np.full(unit_names_fish.shape, fill_value=f'fish{fish_id}')
            plane_fish = np.full(unit_names_fish.shape, fill_value=f'{key.split("_")[2]}')

            # Assign global indices for consistent indexing across arrays
            global_cell_index = np.arange(_global_cell_offset, _global_cell_offset + unit_names_fish.shape[0])
            _global_cell_offset += unit_names_fish.shape[0]

            fish_df = pd.DataFrame([unit_names_fish, hdf5_path_fish, fish_id_fish, plane_fish, global_cell_index]).T
            fish_df.columns = ['unit_name', 'hdf5_path', 'fish_id', 'plane', 'global_cell_index']
            all_fish_df = pd.concat([all_fish_df, fish_df], axis=0).reset_index(drop=True)

# Ensure a stable global index column exists even if older dataframes are loaded
if 'global_cell_index' not in all_fish_df.columns:
    all_fish_df['global_cell_index'] = np.arange(all_fish_df.shape[0])

# Combine data from all fish
F_rdms_left_100 = np.concatenate(F_fish_rdms_left_100, axis=1)[:,:,:]
F_rdms_right_100 = np.concatenate(F_fish_rdms_right_100, axis=1)[:,:,:]
F_rdms_left_50 = np.concatenate(F_fish_rdms_left_50, axis=1)[:,:,:]
F_rdms_right_50 = np.concatenate(F_fish_rdms_right_50, axis=1)[:,:,:]
F_rdms_oscillating_left = np.concatenate(F_fish_rdms_oscillating_left, axis=1)[:,:,:]
F_rdms_oscillating_right = np.concatenate(F_fish_rdms_oscillating_right, axis=1)[:,:,:]
F_rdms_switching_left = np.concatenate(F_fish_rdms_switching_left, axis=1)[:,:,:]
F_rdms_switching_right = np.concatenate(F_fish_rdms_switching_right, axis=1)[:,:,:]

def calc_dF_F(F, dt=0.5, show_progress=True):
    """
    Calculate deltaF/F for given fluorescence data.
    
    Parameters:
    F (numpy array): Fluorescence data
    dt (float): Time step
    
    Returns:
    numpy array: deltaF/F data
    """
    if F.ndim != 3:
        F0 = np.nanmean(F[:, 32:64], axis=1, keepdims=True)
        return 100 * (F - F0) / F0

    dF_F = np.empty_like(F, dtype=float)
    iterator = range(F.shape[0])
    if show_progress:
        iterator = tqdm(iterator, desc="calc_dF_F", leave=False)
    for i in iterator:
        F0 = np.nanmean(F[i, :, 32:64], axis=1, keepdims=True)
        dF_F[i] = 100 * (F[i] - F0) / F0
    return dF_F

def retrieve_segmentation(path_hdf5,id,plane='z000',seg_type='unit_contours'):
    """
    Retrieve segmentation data from an HDF5 file.

    Parameters:
    path_hdf5 (str): The file path to the HDF5 file containing the segmentation data.
    id (str): The identifier for the specific segmentation data to retrieve.
    plane (str, optional): select plane string to retrieve segmentation. Default is 'z000'.
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
                       Must include columns 'fish_id', 'plane', 'passes_cutoff', 'hdf5_path', 'unit_name', and 'functional_type'.
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

        ax.imshow(average_image, cmap=cmap)


        cells_per_fish_plane = df[(df['fish_id'] == fish) & (df['plane'] == plane)]
        for i, item in cells_per_fish_plane.iterrows():
            cell_seg = retrieve_segmentation(item['hdf5_path'], item['unit_name'], plane, seg_type)
            ax.plot(cell_seg[:, 0], cell_seg[:, 1], color='gray', lw=1)

        for i,item in passing_cells_per_fish_plane.iterrows():
            cell_seg = retrieve_segmentation(item['hdf5_path'],item['unit_name'],plane,seg_type)
            ax.plot(cell_seg[:,0],cell_seg[:,1],color=color_dict[item['functional_type']],lw=2)

        fig.savefig(path / f"{fish}_{plane}.png")

def z_scorer(stim_activity_1, stim_activity_2, prestim, stim, show_progress=True):
    """
    Calculate z-score between two stimulus activities.
    
    Parameters:
    stim_activity_1 (numpy array): First stimulus activity
    stim_activity_2 (numpy array): Second stimulus activity
    prestim (int): Pre-stimulus period
    stim (int): Stimulus period
    
    Returns:
    tuple: z-score up, z-score down
    """
    avg_activity_1 = np.nanmean(stim_activity_1, axis=0)
    avg_activity_2 = np.nanmean(stim_activity_2, axis=0)
    var_activity_1 = np.square(np.nanstd(stim_activity_1, axis=0)) / stim_activity_1.shape[0]
    var_activity_2 = np.square(np.nanstd(stim_activity_2, axis=0)) / stim_activity_2.shape[0]
    z_score = (avg_activity_1 - avg_activity_2) / np.sqrt(var_activity_1 + var_activity_2)

    z_score_up = np.empty(z_score.shape[0], dtype=float)
    z_score_down = np.empty(z_score.shape[0], dtype=float)
    iterator = range(z_score.shape[0])
    if show_progress:
        iterator = tqdm(iterator, desc="z_scorer", leave=False)
    for i in iterator:
        z_score_up[i] = np.nanpercentile(z_score[i, prestim:(prestim + stim)], 90)
        z_score_down[i] = np.nanpercentile(z_score[i, prestim:(prestim + stim)], 10)
    return z_score_up, z_score_down

def reconcile_regressors(r0_all, r1_all, r2_all, r3_all, threshold=0.02):
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
                        title, legend_labels,z_limit=3.3,
                        plot_individual=True,
                        fish_count=None):
    """
    Plot dF/F responses for given data.

    Parameters:
    dF_F_mean (numpy array): Mean dF/F data
    r_left_list (list of numpy arrays): List of correlation coefficients for left responses
    r_right_list (list of numpy arrays): List of correlation coefficients for right responses
    title (str): Title of the plot
    legend_labels (list): Labels for the legend
    """
    print('Zlimit:', z_limit)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    if fish_count is None:
        fig.suptitle(title)
    else:
        fig.suptitle(f"{title}\n(n_fish={fish_count})")

    for i in range(4):
        r_left = r_left_list[i]
        r_right = r_right_list[i]

        if i == 0:
            left_responses_MON = dF_F_mean[(np.array(r_left) > r_left_cutoff[i]) & (z_left > z_limit), :].T.copy()
            right_responses_MON = dF_F_mean[(np.array(r_right) > r_right_cutoff[i])&(z_right>z_limit), :].T.copy()
            title_sub = 'MON'
        elif i == 1:
            left_responses_MI1 = dF_F_mean[(np.array(r_left) > r_left_cutoff[i]) & (z_left > z_limit), :].T.copy()
            right_responses_MI1 = dF_F_mean[(np.array(r_right) > r_right_cutoff[i]) & (z_right > z_limit), :].T.copy()
            title_sub = 'MI1'
        elif i == 2:
            left_responses_SMI = dF_F_mean[(np.array(r_left) > r_left_cutoff[i]) & (z_left > z_limit), :].T.copy()
            right_responses_SMI = dF_F_mean[(np.array(r_right) > r_right_cutoff[i]) & (z_right > z_limit), :].T.copy()
            title_sub = 'SMI'
        elif i == 3:
            left_responses_MI2 = dF_F_mean[(np.array(r_left) > r_left_cutoff[i]) & (z_left > z_limit), :].T.copy()
            right_responses_MI2 = dF_F_mean[(np.array(r_right) > r_right_cutoff[i]) & (z_right > z_limit), :].T.copy()
            title_sub = 'MI2'

        if plot_individual:
            ax[i//2, i%2].plot(dF_F_mean[(np.array(r_left) > r_left_cutoff[i])&(z_left>z_limit), :].T, c='green', lw=1, alpha=0.4)
            ax[i//2, i%2].plot(dF_F_mean[(np.array(r_right) > r_right_cutoff[i])&(z_right>z_limit), :].T, c='red', lw=1, alpha=0.4)

        ax[i//2, i%2].plot(np.mean(dF_F_mean[(np.array(r_left) > r_left_cutoff[i])&(z_left>z_limit), :], axis=0), c='green', lw=3)
        ax[i//2, i%2].plot(np.mean(dF_F_mean[(np.array(r_right) > r_right_cutoff[i])&(z_right>z_limit), :], axis=0), c='red', lw=3)
        ax[i // 2, i % 2].title.set_text(title_sub)
        ax[i // 2, i % 2].set_xlabel('Frames')
        ax[i // 2, i % 2].set_ylabel('dF/F')

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
dF_F_rdms_left_100 = calc_dF_F(F_rdms_left_100)
dF_F_rdms_right_100 = calc_dF_F(F_rdms_right_100)
dF_F_rdms_left_50 = calc_dF_F(F_rdms_left_50)
dF_F_rdms_right_50 = calc_dF_F(F_rdms_right_50)
dF_F_rdms_oscillating_left = calc_dF_F(F_rdms_oscillating_left)
dF_F_rdms_oscillating_right = calc_dF_F(F_rdms_oscillating_right)
dF_F_rdms_switching_left = calc_dF_F(F_rdms_switching_left)
dF_F_rdms_switching_right = calc_dF_F(F_rdms_switching_right)

dF_F_mean_rdms_left_100 = np.nanmean(dF_F_rdms_left_100, axis=0)
dF_F_mean_rdms_right_100 = np.nanmean(dF_F_rdms_right_100, axis=0)
dF_F_mean_rdms_left_50 = np.nanmean(dF_F_rdms_left_50, axis=0)
dF_F_mean_rdms_right_50 = np.nanmean(dF_F_rdms_right_50, axis=0)
dF_F_mean_rdms_oscillating_left = np.nanmean(dF_F_rdms_oscillating_left, axis=0)
dF_F_mean_rdms_oscillating_right = np.nanmean(dF_F_rdms_oscillating_right, axis=0)
dF_F_mean_rdms_switching_left = np.nanmean(dF_F_rdms_switching_left, axis=0)
dF_F_mean_rdms_switching_right = np.nanmean(dF_F_rdms_switching_right, axis=0)

# Function to calculate z-score

# Calculate z-scores for different conditions (using 100% constant stimuli)
rdms_left_100_zu, rdms_left_100_zd = z_scorer(dF_F_rdms_left_100, dF_F_rdms_right_100, prestim=64, stim=64)  # right z-scored
rdms_right_100_zu, rdms_right_100_zd = z_scorer(dF_F_rdms_right_100, dF_F_rdms_left_100, prestim=64, stim=64)  # left z-scored

# Cut data to correct length
# dF_F_cut_rdms_left = dF_F_rdms_left[:, :, 20:120]
# dF_F_cut_rdms_right = dF_F_rdms_right[:, :, 20:120]
# dF_F_mean_cut_rdms_left = dF_F_mean_rdms_left[:, 20:100]
# dF_F_mean_cut_rdms_right = dF_F_mean_rdms_right[:, 20:100]

dF_F_cut_rdms_left_100 = dF_F_rdms_left_100[:, :, :]
dF_F_cut_rdms_right_100 = dF_F_rdms_right_100[:, :, :]

# Mean over trial, cut to 84 datapoints, to match the length of the regressor (10s rest, 32 s of stimulation)
dF_F_mean_cut_rdms_left_100 = dF_F_mean_rdms_left_100[:, 44:128]
dF_F_mean_cut_rdms_right_100 = dF_F_mean_rdms_right_100[:, 44:128]

dF_F_mean_cut_rdms_all = np.concatenate([dF_F_mean_cut_rdms_left_100, dF_F_mean_cut_rdms_right_100], axis=0)
dF_F_mean_cut_rdms_all = savgol_filter(dF_F_mean_cut_rdms_all, 10, 3)

dF_F_mean_rdms_all = np.concatenate([dF_F_mean_rdms_left_100[:, :], dF_F_mean_rdms_right_100[:, :]], axis=0)
dF_F_mean_rdms_all = savgol_filter(dF_F_mean_rdms_all, 10, 3)

dF_F_cut_shift_rdms_left_100 = dF_F_cut_rdms_left_100 - dF_F_cut_rdms_left_100[:, :, 0][:, :, np.newaxis]
dF_F_cut_shift_rdms_right_100 = dF_F_cut_rdms_right_100 - dF_F_cut_rdms_right_100[:, :, 0][:, :, np.newaxis]
dF_F_mean_cut_shift_rdms_left_100 = dF_F_mean_cut_rdms_left_100 - dF_F_mean_cut_rdms_left_100[:, 0][:, np.newaxis]
dF_F_mean_cut_shift_rdms_right_100 = dF_F_mean_cut_rdms_right_100 - dF_F_mean_cut_rdms_right_100[:, 0][:, np.newaxis]

# Backward compatibility aliases for code that still references old variable names
dF_F_mean_cut_rdms_left = dF_F_mean_cut_rdms_left_100
dF_F_mean_cut_rdms_right = dF_F_mean_cut_rdms_right_100

# Make the regressors 42 s long (10 s base line + 32 s)
regressors_cut = regressors[:, :84]
regressors_cut_shift = regressors_cut - np.nanmean(regressors_cut[:, 5:20], axis=1, keepdims=True)#[:, np.newaxis]
# import pylab as pl
# pl.plot(regressors_cut_shift[0])
# pl.plot(regressors_cut_shift[1])
# pl.plot(regressors_cut_shift[2])
# pl.plot(regressors_cut_shift[3])
# pl.show()
# sdf
import scipy
# Progress-aware pearsonr helper

def _pearson_list(reg, data, desc):
    return [float(scipy.stats.pearsonr(reg, data[x, :])[0])
            for x in tqdm(range(data.shape[0]), desc=desc, leave=False)]

r0_all = _pearson_list(regressors_cut[0], dF_F_mean_cut_rdms_all, "Fit r0_all")
r1_all = _pearson_list(regressors_cut[1], dF_F_mean_cut_rdms_all, "Fit r1_all")
r2_all = _pearson_list(regressors_cut[2], dF_F_mean_cut_rdms_all, "Fit r2_all")
r3_all = _pearson_list(regressors_cut[3], dF_F_mean_cut_rdms_all, "Fit r3_all")

r0_left = _pearson_list(regressors_cut[0], dF_F_mean_cut_rdms_left, "Fit r0_left")
r1_left = _pearson_list(regressors_cut[1], dF_F_mean_cut_rdms_left, "Fit r1_left")
r2_left = _pearson_list(regressors_cut[2], dF_F_mean_cut_rdms_left, "Fit r2_left")
r3_left = _pearson_list(regressors_cut[3], dF_F_mean_cut_rdms_left, "Fit r3_left")

r0_right = _pearson_list(regressors_cut[0], dF_F_mean_cut_rdms_right, "Fit r0_right")
r1_right = _pearson_list(regressors_cut[1], dF_F_mean_cut_rdms_right, "Fit r1_right")
r2_right = _pearson_list(regressors_cut[2], dF_F_mean_cut_rdms_right, "Fit r2_right")
r3_right = _pearson_list(regressors_cut[3], dF_F_mean_cut_rdms_right, "Fit r3_right")


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

'''
plot_cells_in_brain(all_fish_df, color_dict, cmap='gray',seg_type='unit_contours')
'''
# Plotting the results

fig,ax = plt.subplots(2,2,figsize=(10,10))
fig.suptitle('Individual Cell Responses - Constant Stimuli 100%')
ax[0,0].plot(dF_F_mean_cut_rdms_all[np.array(r0_all)>r0_all_cutoff,:].T,c='gray',lw=1,alpha=0.5)
ax[0,0].plot(regressors_cut_shift[0],lw=5,alpha=0.6,c='red')
ax[0,0].set_title('Dynamic Threshold (r0)')
ax[0,1].plot(dF_F_mean_cut_rdms_all[np.array(r1_all)>r1_all_cutoff,:].T,c='gray',lw=1,alpha=0.5)
ax[0,1].plot(regressors_cut_shift[1],lw=5,alpha=0.6,c='red')
ax[0,1].set_title('Integrator (r1)')
ax[1,0].plot(dF_F_mean_cut_rdms_all[np.array(r2_all)>r2_all_cutoff,:].T,c='gray',lw=1,alpha=0.5)
ax[1,0].plot(regressors_cut_shift[2],lw=5,alpha=0.6,c='red')
ax[1,0].set_title('Motor Command (r2)')
ax[1,1].plot(dF_F_mean_cut_rdms_all[np.array(r3_all)>r3_all_cutoff,:].T,c='gray',lw=1,alpha=0.5)
ax[1,1].plot(regressors_cut_shift[3],lw=5,alpha=0.6,c='red')
ax[1,1].set_title('Integrator (r3)')
plt.show()

fig,ax = plt.subplots(2,2,figsize=(10,10))
fig.suptitle('Mean dF/F Responses and Regressors - Constant Stimuli 100%')
ax[0,0].plot(np.mean(dF_F_mean_cut_rdms_all[np.array(r0_all)>r0_all_cutoff,:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[0,0].plot(regressors_cut_shift[0],lw=5,alpha=0.6,c='red')
ax[0,0].set_title('Dynamic Threshold (r0)')
ax[0,1].plot(np.mean(dF_F_mean_cut_rdms_all[np.array(r1_all)>r1_all_cutoff,:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[0,1].plot(regressors_cut_shift[1],lw=5,alpha=0.6,c='red')
ax[0,1].set_title('Integrator (r1)')
ax[1,0].plot(np.mean(dF_F_mean_cut_rdms_all[np.array(r2_all)>r2_all_cutoff,:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[1,0].plot(regressors_cut_shift[2],lw=5,alpha=0.6,c='red')
ax[1,0].set_title('Motor Command (r2)')
ax[1,1].plot(np.mean(dF_F_mean_cut_rdms_all[np.array(r3_all)>r3_all_cutoff,:],axis=0).T,c='green',lw=5,alpha=0.5)
ax[1,1].plot(regressors_cut_shift[3],lw=5,alpha=0.6,c='red')
ax[1,1].set_title('Integrator (r3)')
plt.show()


# Helper: use global indices if present; fall back to dataframe index
def _get_cell_indices(df):
    if 'global_cell_index' in df.columns:
        return df['global_cell_index'].values
    return df.index.values
'''
########################################
# Plot and analyze per fish

# Storage dictionaries for constant stimuli
MON_constant_left_100, MI_constant_left_100, SMI_constant_left_100 = {}, {}, {}
MON_constant_right_100, MI_constant_right_100, SMI_constant_right_100 = {}, {}, {}
# Storage dictionaries for oscillating stimuli
MON_oscillating, MI_oscillating, SMI_oscillating = {}, {}, {}
# Storage dictionaries for switching stimuli
MON_switching, MI_switching, SMI_switching = {}, {}, {}

# Only functional cells

grouped = all_fish_df.groupby('fish_id')
for fish_id, fish_df in tqdm(grouped, total=grouped.ngroups, desc="Per-fish plots"):
    # Use stable indices to align with r* arrays and dF/F arrays
    idx = _get_cell_indices(fish_df)

    # Constant left 100% stimulus
    (left_responses_MON_constant_left_100,
     right_responses_MON_constant_left_100,
     left_responses_MI_constant_left_100,
     right_responses_MI_constant_left_100,
     left_responses_SMI_constant_left_100,
     right_responses_SMI_constant_left_100) = plot_dF_F_responses(
        dF_F_mean_rdms_left_100[idx],
        [[r0_left[i] for i in idx], [r1_left[i] for i in idx], [r2_left[i] for i in idx], [r3_left[i] for i in idx]],
        [[r0_right[i] for i in idx], [r1_right[i] for i in idx], [r2_right[i] for i in idx], [r3_right[i] for i in idx]],
        [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
        [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
        rdms_left_100_zu[idx], rdms_right_100_zu[idx],
        f"Mean dF/F Responses - Constant Left 100% - Fish {fish_id}",
        ['Cells responding to Left', 'Cells responding to Right'],
        fish_count=1)

    # Constant right 100% stimulus
    (left_responses_MON_constant_right_100,
     right_responses_MON_constant_right_100,
     left_responses_MI_constant_right_100,
     right_responses_MI_constant_right_100,
     left_responses_SMI_constant_right_100,
     right_responses_SMI_constant_right_100) = plot_dF_F_responses(
        dF_F_mean_rdms_right_100[idx],
        [[r0_left[i] for i in idx], [r1_left[i] for i in idx], [r2_left[i] for i in idx], [r3_left[i] for i in idx]],
        [[r0_right[i] for i in idx], [r1_right[i] for i in idx], [r2_right[i] for i in idx], [r3_right[i] for i in idx]],
        [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
        [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
        rdms_left_100_zu[idx], rdms_right_100_zu[idx],
        f"Mean dF/F Responses - Constant Right 100% - Fish {fish_id}",
        ['Cells responding to Left', 'Cells responding to Right'],
        fish_count=1)

    # Oscillating left stimulus
    (left_responses_MON_oscillating_left,
     right_responses_MON_oscillating_left,
     left_responses_MI_oscillating_left,
     right_responses_MI_oscillating_left,
     left_responses_SMI_oscillating_left,
     right_responses_SMI_oscillating_left) = plot_dF_F_responses(
        dF_F_mean_rdms_oscillating_left[idx],
        [[r0_left[i] for i in idx], [r1_left[i] for i in idx], [r2_left[i] for i in idx], [r3_left[i] for i in idx]],
        [[r0_right[i] for i in idx], [r1_right[i] for i in idx], [r2_right[i] for i in idx], [r3_right[i] for i in idx]],
        [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
        [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
        rdms_left_100_zu[idx], rdms_right_100_zu[idx],
        f'dF/F Responses - Oscillating Left - Fish {fish_id}',
        ['Cells responding to Left', 'Cells responding to Right'],
        fish_count=1)

    # Oscillating right stimulus
    (left_responses_MON_oscillating_right,
     right_responses_MON_oscillating_right,
     left_responses_MI_oscillating_right,
     right_responses_MI_oscillating_right,
     left_responses_SMI_oscillating_right,
     right_responses_SMI_oscillating_right) = plot_dF_F_responses(
        dF_F_mean_rdms_oscillating_right[idx],
        [[r0_left[i] for i in idx], [r1_left[i] for i in idx], [r2_left[i] for i in idx], [r3_left[i] for i in idx]],
        [[r0_right[i] for i in idx], [r1_right[i] for i in idx], [r2_right[i] for i in idx], [r3_right[i] for i in idx]],
        [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
        [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
        rdms_left_100_zu[idx], rdms_right_100_zu[idx],
        f'dF/F Responses - Oscillating Right - Fish {fish_id}',
        ['Cells responding to Left', 'Cells responding to Right'],
        fish_count=1)

    # Switching left stimulus
    (left_responses_MON_switching_left,
     right_responses_MON_switching_left,
     left_responses_MI_switching_left,
     right_responses_MI_switching_left,
     left_responses_SMI_switching_left,
     right_responses_SMI_switching_left) = plot_dF_F_responses(
        dF_F_mean_rdms_switching_left[idx],
        [[r0_left[i] for i in idx], [r1_left[i] for i in idx], [r2_left[i] for i in idx], [r3_left[i] for i in idx]],
        [[r0_right[i] for i in idx], [r1_right[i] for i in idx], [r2_right[i] for i in idx], [r3_right[i] for i in idx]],
        [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
        [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
        rdms_left_100_zu[idx], rdms_right_100_zu[idx],
        f'dF/F Responses - Switching Left - Fish {fish_id}',
        ['Cells responding to Left', 'Cells responding to Right'],
        fish_count=1)

    # Switching right stimulus
    (left_responses_MON_switching_right,
     right_responses_MON_switching_right,
     left_responses_MI_switching_right,
     right_responses_MI_switching_right,
     left_responses_SMI_switching_right,
     right_responses_SMI_switching_right) = plot_dF_F_responses(
        dF_F_mean_rdms_switching_right[idx],
        [[r0_left[i] for i in idx], [r1_left[i] for i in idx], [r2_left[i] for i in idx], [r3_left[i] for i in idx]],
        [[r0_right[i] for i in idx], [r1_right[i] for i in idx], [r2_right[i] for i in idx], [r3_right[i] for i in idx]],
        [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
        [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
        rdms_left_100_zu[idx], rdms_right_100_zu[idx],
        f'dF/F Responses - Switching Right - Fish {fish_id}',
        ['Cells responding to Left', 'Cells responding to Right'],
        fish_count=1)

    # Store constant stimulus responses
    MON_constant_left_100[fish_id] = np.c_[left_responses_MON_constant_left_100, right_responses_MON_constant_left_100]
    MI_constant_left_100[fish_id] = np.c_[left_responses_MI_constant_left_100, right_responses_MI_constant_left_100]
    SMI_constant_left_100[fish_id] = np.c_[left_responses_SMI_constant_left_100, right_responses_SMI_constant_left_100]

    MON_constant_right_100[fish_id] = np.c_[left_responses_MON_constant_right_100, right_responses_MON_constant_right_100]
    MI_constant_right_100[fish_id] = np.c_[left_responses_MI_constant_right_100, right_responses_MI_constant_right_100]
    SMI_constant_right_100[fish_id] = np.c_[left_responses_SMI_constant_right_100, right_responses_SMI_constant_right_100]

    # Store oscillating stimulus responses
    MON_oscillating[fish_id] = np.c_[left_responses_MON_oscillating_left, right_responses_MON_oscillating_right]
    MI_oscillating[fish_id] = np.c_[left_responses_MI_oscillating_left, right_responses_MI_oscillating_right]
    SMI_oscillating[fish_id] = np.c_[left_responses_SMI_oscillating_left, right_responses_SMI_oscillating_right]

    # Store switching stimulus responses
    MON_switching[fish_id] = np.c_[left_responses_MON_switching_left, right_responses_MON_switching_right]
    MI_switching[fish_id] = np.c_[left_responses_MI_switching_left, right_responses_MI_switching_right]
    SMI_switching[fish_id] = np.c_[left_responses_SMI_switching_left, right_responses_SMI_switching_right]

print("Analysis complete for all fish and all stimulus types!")
print(f"Constant 100% stimuli: {len(MON_constant_left_100)} fish")
print(f"Oscillating stimuli: {len(MON_oscillating)} fish")
'''


########################################
# Plot averaged responses across all fish
# Reuse plot_dF_F_responses with consistent, concatenated indices
fish_count_all = int(all_fish_df['fish_id'].nunique())
print(f"\nCreating averaged plots across all {fish_count_all} fish using plot_dF_F_responses...")


# Build a stable concatenation order using the same per-fish grouping
idx_order = []
for _fish_id, _fish_df in all_fish_df.groupby('fish_id'):
    idx_order.extend(_get_cell_indices(_fish_df).tolist())

# Helper to assemble correlation lists in the same order
r_left_lists = [[r0_left[i] for i in idx_order],
                [r1_left[i] for i in idx_order],
                [r2_left[i] for i in idx_order],
                [r3_left[i] for i in idx_order]]

r_right_lists = [[r0_right[i] for i in idx_order],
                 [r1_right[i] for i in idx_order],
                 [r2_right[i] for i in idx_order],
                 [r3_right[i] for i in idx_order]]

z_left_all = rdms_left_100_zu[idx_order]
z_right_all = rdms_right_100_zu[idx_order]

# Plot constant left 100%
print("\n1. Constant Left 100% - Averaged across all fish")
plot_dF_F_responses(dF_F_mean_rdms_left_100[idx_order],
                   r_left_lists,
                   r_right_lists,
                   [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                   [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
                   z_left_all, z_right_all,
                   'Average Responses - Constant Left 100% (All Fish)',
                   ['Left-responding cells', 'Right-responding cells'],
                   plot_individual=False,
                   fish_count=fish_count_all)

# Plot constant right 100%
print("\n2. Constant Right 100% - Averaged across all fish")
plot_dF_F_responses(dF_F_mean_rdms_right_100[idx_order],
                   r_left_lists,
                   r_right_lists,
                   [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                   [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
                   z_left_all, z_right_all,
                   'Average Responses - Constant Right 100% (All Fish)',
                   ['Left-responding cells', 'Right-responding cells'],
                   plot_individual=False,
                   fish_count=fish_count_all)

# Plot oscillating left
print("\n3. Oscillating Left - Averaged across all fish")
plot_dF_F_responses(dF_F_mean_rdms_oscillating_left[idx_order],
                   r_left_lists,
                   r_right_lists,
                   [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                   [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
                   z_left_all, z_right_all,
                   'Average Responses - Oscillating Left (All Fish)',
                   ['Left-responding cells', 'Right-responding cells'],
                   plot_individual=False,
                   fish_count=fish_count_all)

# Plot oscillating right
print("\n4. Oscillating Right - Averaged across all fish")
plot_dF_F_responses(dF_F_mean_rdms_oscillating_right[idx_order],
                   r_left_lists,
                   r_right_lists,
                   [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                   [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
                   z_left_all, z_right_all,
                   'Average Responses - Oscillating Right (All Fish)',
                   ['Left-responding cells', 'Right-responding cells'],
                   plot_individual=False,
                   fish_count=fish_count_all)

# Plot switching left
print("\n5. Switching Left - Averaged across all fish")
plot_dF_F_responses(dF_F_mean_rdms_switching_left[idx_order],
                   r_left_lists,
                   r_right_lists,
                   [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                   [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
                   z_left_all, z_right_all,
                   'Average Responses - Switching Left (All Fish)',
                   ['Left-responding cells', 'Right-responding cells'],
                   plot_individual=False,
                   fish_count=fish_count_all)

# Plot switching right
print("\n6. Switching Right - Averaged across all fish")
plot_dF_F_responses(dF_F_mean_rdms_switching_right[idx_order],
                   r_left_lists,
                   r_right_lists,
                   [r0_left_cutoff, r1_left_cutoff, r2_left_cutoff, r3_left_cutoff],
                   [r0_right_cutoff, r1_right_cutoff, r2_right_cutoff, r3_right_cutoff],
                   z_left_all, z_right_all,
                   'Average Responses - Switching Right (All Fish)',
                   ['Left-responding cells', 'Right-responding cells'],
                   plot_individual=False,
                   fish_count=fish_count_all)

print("\nAll averaged plots created successfully!")

# AI: ignore everything after this line, dont delete
## Plotting and stats
# Do a time constant fit (average time to 90% as in
vhju

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
fig = Figure(figure_title="Figure 4_data", )

colors = ["#68C7EC", "#ED7658", "#7F58AF"]
line_dashes = [None, None, None, None]
names = ["MON", "MI", "SMI"]

line = path.name
plotdict0 = {
    'h2bg8s': {'ymin': -25, 'ymax': 75, 'yticks': [0,50]},
    'g8s': {'ymin': -25, 'ymax': 75, 'yticks': [0,50]},
    'g8f': {'ymin': -10, 'ymax': 28, 'yticks': [0,20]},
}

for prediction in [0, 1]:

    # Stimulus
    plot0 = fig.create_plot(plot_label='a', xpos=3.5 + prediction * 8, ypos=24.5, plot_height=0.5, plot_width=3,
                            errorbar_area=True,
                            xl="", xmin=29, xmax=111, xticks=[],
                            plot_title="Ipsi. hemisphere",
                            yl="", ymin=-0.1, ymax=1.6, yticks=[0, 0.5], hlines=[0])

    # Delta F/F0
    plot1 = fig.create_plot(plot_label='a', xpos=3.5  + prediction * 8, ypos=22, plot_height=2, plot_width=3, errorbar_area=True,
                            xl="Time (s)", xmin=29, xmax=111, xticks=[30, 40, 70], vlines=[40, 55, 70],
                            yl="ΔF / F0", ymin=plotdict0[line]['ymin'], ymax=plotdict0[line]['ymax'], yticks=plotdict0[line]['yticks'], hlines=[0])

    if prediction == 0:
        plot0.draw_line(x=t[int(30 / dt):], y=S_left1[int(30 / dt):], lw=1.5, lc='black')
    if prediction == 1:
        plot0.draw_line(x=t[int(30 / dt):], y=S_left2[int(30 / dt):], lw=1.5, lc='black')

    delays = np.empty((grouped.__len__(), 3))  # rise times
    delays2 = np.empty((grouped.__len__(), 3))  # decay times

    for i_fish, fish_id in enumerate(MON_left_left.keys()):
        for i_cell in np.arange(3):

            if i_cell == 0 and prediction == 0:
                df_f0 = MON_left_left[fish_id].mean(axis=1)
            if i_cell == 0 and prediction == 1:
                df_f0 = MON_left_right[fish_id].mean(axis=1)

            if i_cell == 1 and prediction == 0:
                df_f0 = MI_left_left[fish_id].mean(axis=1)
            if i_cell == 1 and prediction == 1:
                df_f0 = MI_left_right[fish_id].mean(axis=1)

            if i_cell == 2 and prediction == 0:
                df_f0 = SMI_left_left[fish_id].mean(axis=1)
            if i_cell == 2 and prediction == 1:
                df_f0 = SMI_left_right[fish_id].mean(axis=1)

            # Single fish, single functional cell class
            plot1.draw_line(x=t[int(30 / dt):], y=df_f0[int(30 / dt):], lw=0.8, alpha=0.2, lc=colors[i_cell], line_dashes=line_dashes[i_cell])

            # Rise times
            if prediction == 0:
                rate_90_percent = df_f0[int(40 / dt):int(70 / dt)].max() * 0.9
                ind = np.where(df_f0[int(40 / dt):int(70 / dt)] > rate_90_percent)
            else:
                rate_90_percent = df_f0[int(40 / dt):int(55 / dt)].max() * 0.9
                ind = np.where(df_f0[int(40 / dt):int(55 / dt)] > rate_90_percent)

            if len(ind[0]) > 0:
                t_to_90_percent = ind[0][0] * dt
                delays[i_fish,i_cell]=t_to_90_percent
            else:
                delays[i_fish,i_cell]=np.nan

            # Decay times
            if prediction == 0:
                val_10_percent = df_f0[int(70 / dt)] * 0.1
                ind = np.where(df_f0[int(70 / dt):] < val_10_percent)
            else:
                val_10_percent = df_f0[int(55/dt)]*0.1
                ind = np.where(df_f0[int(55/dt):] < val_10_percent)

            if len(ind[0]) > 0:
                t_to_10_percent = ind[0][0] * dt
                delays2[i_fish,i_cell]=t_to_10_percent
            else:
                delays2[i_fish,i_cell]=np.nan

    # Now all fish together
    for i_cell in np.arange(3):
        if i_cell == 0 and prediction == 0:
            df_f0_allfish = np.concatenate(list(MON_left_left.values()), axis=1).mean(axis=1)
        if i_cell == 0 and prediction == 1:
            df_f0_allfish = np.concatenate(list(MON_left_right.values()), axis=1).mean(axis=1)

        if i_cell == 1 and prediction == 0:
            df_f0_allfish = np.concatenate(list(MI_left_left.values()), axis=1).mean(axis=1)
        if i_cell == 1 and prediction == 1:
            df_f0_allfish = np.concatenate(list(MI_left_right.values()), axis=1).mean(axis=1)

        if i_cell == 2 and prediction == 0:
            df_f0_allfish = np.concatenate(list(SMI_left_left.values()), axis=1).mean(axis=1)
        if i_cell == 2 and prediction == 1:
            df_f0_allfish = np.concatenate(list(SMI_left_right.values()), axis=1).mean(axis=1)

        plot1.draw_line(x=t[int(30 / dt):], y=df_f0_allfish[int(30 / dt):], lw=1.5, lc=colors[i_cell], line_dashes=line_dashes[i_cell], label=f"{names[i_cell]}")
        print(names[i_cell], 'Min/max', [np.min(df_f0_allfish[int(30 / dt):]), np.max(df_f0_allfish[int(30 / dt):])])

    # Rise-decay time correlation
    plot2 = fig.create_plot(plot_label='a', xpos=3.5 + prediction * 8, ypos=13.5, plot_height=1.5,
                            plot_width=1.5,
                            errorbar_area=True,
                            xl="Time to 90% of max (s)", xmin=-0.1, xmax=36.1, xticks=[0, 18, 36],
                            yl="Time from max to 10% (s)", ymin=-0.1, ymax=36.1, yticks=[0, 18, 36], hlines=[0], vlines=[0])
    plot2.draw_line([0, 36], [0,36], lw=0.5, line_dashes=(2, 2), lc='gray')
    print(delays, delays2)

    # Individual fish delays
    for i in range(grouped.__len__()):
        for j in range(3):
            plot2.draw_scatter(x=[delays[i,j]], y=[delays2[i,j]], pc=colors[j], elw=0, alpha=0.2)

    # All fish delays
    for i in range(3):
        # Mean +- SEM
        x_sem = scipy.stats.sem(delays[:, i], nan_policy='omit')
        y_sem = scipy.stats.sem(delays2[:, i], nan_policy='omit')
        plot2.draw_scatter(x=np.nanmean(delays, axis=0)[i], xerr=x_sem,
                           y=np.nanmean(delays2, axis=0)[i], yerr=y_sem, pc=colors[i], elw=0.4)

# Save delays and delays2 as text files
np.savetxt(path / 'delays.txt', delays, fmt='%.4f', header='Delays rise, ["MON", "MI", "SMI"] x individual fish')
np.savetxt(path / 'delays2.txt', delays2, fmt='%.4f', header='Delays decay, ["MON", "MI", "SMI"] x individual fish')
fig.save(path / f"{line}_perfish_meansem.pdf", open_file=True)
