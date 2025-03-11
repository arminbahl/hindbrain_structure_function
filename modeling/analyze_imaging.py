import h5py
import numpy as np
from numba import jit
import numpy as np
import pylab as pl
import pathlib

# matplotlib.use("macosx")
import matplotlib.colors as colors
import numpy as np
import scipy

from analysis_helpers.analysis.utils.figure_helper import Figure

fp = h5py.File('/Users/arminbahl/Nextcloud/CLEM_paper_data/clem_zfish1/activity_recordings/untitled folder/dF_F_dynamics.hdf5', 'r')
MON_leftward = fp["all/dF_F_mean_rdms_left/dynamic_threshold"]
MON_leftward_then_rightward = fp["all/dF_F_mean_rdms_left_right/dynamic_threshold"]
MON_rightward = fp["all/dF_F_mean_rdms_right/dynamic_threshold"]
MON_rightward_then_leftward = fp["all/dF_F_mean_rdms_right_left/dynamic_threshold"]
time = np.arange(0, 70, 0.5)

for i in range(MON_leftward.shape[0]):
    pl.plot(MON_leftward[i])
pl.show()
