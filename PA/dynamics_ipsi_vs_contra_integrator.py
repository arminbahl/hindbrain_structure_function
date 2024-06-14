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

def refine_figure(fig,ax,fig_name="no name figure"):
    # Overlay shaded rectangle for stimulus epoch



    # Set font of legend text to Arial
    legend = plt.legend(frameon=False, loc='upper left', fontsize='small')
    for text in legend.get_texts():
        text.set_fontfamily('Arial')
    # Set aspect ratio to 1
    ratio = 1.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)




path_to_data = get_base_path()
pa_table = load_pa_table(path_to_data.joinpath("paGFP").joinpath("photoactivation_cells_table.csv"))
fiji_dynamics=True

cell_type_categories = {'morphology':['ipsilateral','contralateral'],
                            'neurotransmitter':['inhibitory','excitatory'],
                            'function':['integrator','dynamic_threshold','dynamic threshold','motor_command','motor command']}

for i, cell in pa_table.iterrows():
    if type(cell.cell_type_labels) == list:
        for label in cell.cell_type_labels:
            if label in cell_type_categories['morphology']:
                pa_table.loc[i, 'morphology'] = label
            elif label in cell_type_categories['function']:
                pa_table.loc[i, 'function'] = label
            elif label in cell_type_categories['neurotransmitter']:
                pa_table.loc[i, 'neurotransmitter'] = label



fig_ipsi, ax_ipsi = plt.subplots()
fig_contra, ax_contra = plt.subplots()
all_ipsi = None
all_contra = None




for i,cell in pa_table.iterrows():
    path_to_file = path_to_data / 'paGFP' / cell.cell_name / (cell.cell_name + "_dynamics.hdf5")

    with h5py.File(path_to_file, 'r') as f:
        smooth_avg_activity_left = list(f['dF_F']['average_dots_left'])
        smooth_avg_activity_right = list(f['dF_F']['average_dots_right'])

    smooth_avg_activity_right = smooth_avg_activity_right/max(smooth_avg_activity_right)
    smooth_avg_activity_left = smooth_avg_activity_left / max(smooth_avg_activity_left)

    #set priors
    dt = 0.5
    time_axis = np.arange(len(smooth_avg_activity_right)) * dt

    #plot
    if cell['function'] == 'integrator' and cell['morphology'] == 'ipsilateral':
        ax_ipsi.plot(time_axis, smooth_avg_activity_left, color='#%02x%02x%02x' %(254, 179, 38), alpha=0.7, lw=1, label='Smoothed Average Left')
        # ax_ipsi.plot(time_axis, smooth_avg_activity_right, color='red', alpha=0.7, linestyle='--', lw=1, label='Smoothed Average Right')

        if all_ipsi is None:
            all_ipsi = smooth_avg_activity_left
        else:
            all_ipsi = np.vstack((all_ipsi, smooth_avg_activity_left))

    if cell['function'] == 'integrator' and cell['morphology'] == 'contralateral':
        ax_contra.plot(time_axis, smooth_avg_activity_left, color='#%02x%02x%02x' %(232, 77, 138), alpha=0.7, lw=1, label='Smoothed Average Left')
        # ax_contra.plot(time_axis, smooth_avg_activity_right, color='red', alpha=0.7, linestyle='--', lw=1, label='Smoothed Average Right')
        if all_contra is None:
            all_contra = smooth_avg_activity_left
        else:
            all_contra = np.vstack((all_contra, smooth_avg_activity_left))



ax_ipsi.axvspan(10, 50, color='gray', alpha=0.1, label='Stimulus Epoch')
ax_contra.axvspan(10, 50, color='gray', alpha=0.1, label='Stimulus Epoch')


ax_ipsi.set_title(f'Average and Individual Trial Activity Dynamics\n Ipsilateral integrators')
ax_contra.set_title(f'Average and Individual Trial Activity Dynamics\n Contralateral integrators')

ax_ipsi.set_xlabel('Time (seconds)')
ax_contra.set_xlabel('Time (seconds)')

ax_ipsi.set_ylabel('Activity')
ax_contra.set_ylabel('Activity')

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='b', lw=1, label='Smoothed Average Left', alpha=0.7),]
                   # Line2D([0], [0], color='r', lw=1, linestyle='--', label='Smoothed Average Right', alpha=0.7)]

legend_ipsi = ax_ipsi.legend(frameon=False, loc='upper left', fontsize='small',handles=legend_elements)
legend_contra = ax_contra.legend(frameon=False, loc='upper left', fontsize='small',handles=legend_elements)

for text in legend_ipsi.get_texts():
    text.set_fontfamily('Arial')

for text in legend_contra.get_texts():
    text.set_fontfamily('Arial')


ratio = 1.0
x_left_i, x_right_i = ax_ipsi.get_xlim()
y_low_i, y_high_i = ax_ipsi.get_ylim()

x_left_c, x_right_c = ax_contra.get_xlim()
y_low_c, y_high_c = ax_contra.get_ylim()

low = np.min([y_low_i,y_low_c])
high = np.max([y_high_i,y_high_c])

ax_ipsi.set_ylim(ax_contra.get_ylim())
ax_contra.set_aspect(abs((x_right_c - x_left_c) / (low - high)) * ratio)
ax_ipsi.set_aspect(abs((x_right_i - x_left_i) / (low - high)) * ratio)


os.makedirs(path_to_data / 'make_figures_FK_output' / 'dynamics_comparison', exist_ok=True)
fig_ipsi.savefig(path_to_data / 'make_figures_FK_output' / 'dynamics_comparison' / 'ipsilateral_integrator.pdf',dpi=600)
fig_contra.savefig(path_to_data / 'make_figures_FK_output' / 'dynamics_comparison' / 'contralateral_integrator.pdf',dpi=600)
fig_ipsi.savefig(path_to_data / 'make_figures_FK_output' / 'dynamics_comparison' / 'ipsilateral_integrator.png',dpi=600)
fig_contra.savefig(path_to_data / 'make_figures_FK_output' / 'dynamics_comparison' / 'contralateral_integrator.png',dpi=600)

