from analysis_helpers.analysis.two_photon_modules.segmented_cell_activity_viewer import *

# Path to experiment folder
path = r"W:\M11 2P microscopes\Kim\ARMIN\dot_motion_coherence_opposing\2025-03-03_19-56-46_fish004_setup0_arena0"
#path = r"W:\M11 2P microscopes\Kim\ARMIN\dot_motion_coherence_opposing\2025-03-03_19-57-02_fish005_setup1_arena0"

cv = SegmentedCellActivityViewer(
    path_to_data = path,
    stimulus_start=int(10/0.5), # in timesteps
    stimulus_end=int(30/0.5), # in timesteps
    z_planes=0,
)

# Process data with segmentation
cv.load_and_process_data()
# Show data
cv.show_plot()
