import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from analysis_helpers.analysis.utils.figure_helper import *
import h5py
import nrrd
from scipy.interpolate import interp1d


def extract_traces_from_roi(experimentname):
    f_raw = h5py.File(
        rf"Z:\Florian\function-neurotransmitter-morphology\functional\{experimentname}\{experimentname}_raw_data.hdf5")
    f_processed = h5py.File(
        rf"Z:\Florian\function-neurotransmitter-morphology\functional\{experimentname}\{experimentname}_preprocessed_data.h5")
    roi_nrrd, header = nrrd.read(
        rf"Z:\Florian\function-neurotransmitter-morphology\functional\{experimentname}\{experimentname}_roi.nrrd",
        index_order='C')

    roi = np.where(roi_nrrd != np.min(roi_nrrd))

    complete_experiment = np.array(f_raw["z_plane0000"]['imaging_green_channel'])

    imaging_time = f_raw[f"z_plane0000/imaging_information"][:, 0]
    stimulus_start_times = f_raw[f"z_plane0000/stimulus_information"][:, 0]
    stimulus_end_times = f_raw[f"z_plane0000/stimulus_information"][:, 1]
    stimulus_start_indices = f_raw[f"z_plane0000/stimulus_information"][:, 2].astype(int)
    complete_experiment = complete_experiment

    # code stolen from katja/armin
    f_imaging_interpolation_function_F = interp1d(imaging_time, complete_experiment, axis=0, bounds_error=False)
    stimulus_aligned_F_sum = dict({})
    stimulus_aligned_F_count = dict({})

    single_stimulus_aligned_responses = []  # List to store single stimulus aligned responses

    for stimulus_start_time, i_stim in zip(stimulus_start_times, stimulus_start_indices):
        print("Stimulus aligning", stimulus_start_time, i_stim)

        if stimulus_start_time > imaging_time[-1]:
            continue

        ts = np.arange(stimulus_start_time,
                       stimulus_start_time + 60 - 0.5 / 2,
                       0.5)

        stimulus_aligned_F = f_imaging_interpolation_function_F(ts).astype(
            np.float64)  # We need to be able to sum a lot

        if i_stim in stimulus_aligned_F_sum:
            stimulus_aligned_F_sum[i_stim] = np.nansum([stimulus_aligned_F_sum[i_stim], stimulus_aligned_F],
                                                       axis=0)
            stimulus_aligned_F_count[i_stim] += (~np.isnan(stimulus_aligned_F)).astype(np.int64)
        else:
            stimulus_aligned_F_sum[i_stim] = stimulus_aligned_F
            stimulus_aligned_F_count[i_stim] = (~np.isnan(stimulus_aligned_F)).astype(np.int64)

        single_stimulus_aligned_responses.append(stimulus_aligned_F)  # Append single stimulus aligned response

    single_stimulus_aligned_responses = np.array(single_stimulus_aligned_responses)

    stimululs_aligned_activity_stimulus0 = single_stimulus_aligned_responses[~stimulus_start_indices.astype(bool), :, :,
                                           :]
    stimululs_aligned_activity_stimulus1 = single_stimulus_aligned_responses[stimulus_start_indices.astype(bool), :, :,
                                           :]

    roi_stimulus0 = stimululs_aligned_activity_stimulus0[:, :, roi[0], roi[1]]
    roi_stimulus1 = stimululs_aligned_activity_stimulus1[:, :, roi[0], roi[1]]

    roi_mean_stimulus0 = np.mean(roi_stimulus0, axis=2)
    roi_mean_stimulus1 = np.mean(roi_stimulus1, axis=2)

    df_f0_roi_stim0 = (roi_mean_stimulus0 - np.mean(roi_mean_stimulus0[:, 5:15], axis=1)[:, np.newaxis]) / np.mean(
        roi_mean_stimulus0[:, 5:15], axis=1)[:, np.newaxis]
    df_f0_roi_stim1 = (roi_mean_stimulus1 - np.mean(roi_mean_stimulus1[:, 5:15], axis=1)[:, np.newaxis]) / np.mean(
        roi_mean_stimulus1[:, 5:15], axis=1)[:, np.newaxis]

    return df_f0_roi_stim0, df_f0_roi_stim1




