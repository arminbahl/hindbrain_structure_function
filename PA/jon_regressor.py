import h5py
import numpy as np
from pathlib import Path
import pylab as pl
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
import re
import navis
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # set variables
    np.set_printoptions(suppress=True)
    width_brain = 495.56
    data_path = get_base_path()

    regressors = np.load(data_path / 'paGFP' /  "regressors_old.npy")
    regressors = regressors[:,:120]
    np.savetxt(data_path / f"regressors_old.txt", regressors, delimiter='\t')

    dt = 0.5

    # Define the pattern
    pattern = re.compile(r'^\d{8}\.\d$')

    cells = os.listdir(data_path / 'paGFP')

    # Filter files based on the pattern
    cells = [f for f in cells if pattern.match(f)]
    df = None
    for directory in cells:
        swc = navis.read_swc(data_path / 'paGFP' / directory / f'{directory}.swc')
        left_hemisphere = swc.nodes.iloc[0]['x'] < width_brain / 2
        temp_path = data_path / 'paGFP' / directory / f'{directory}_dynamics.hdf5'

        if directory in ["20230324.1","20230327.1"]:
            with h5py.File(temp_path, 'r') as f:
                df_F_left_dots_avg = np.array(f['dF_F/average_dots_left'])
                df_F_right_dots_avg = np.array(f['dF_F/average_dots_right'])

            rel_left_dots= np.nan
            rel_right_dots= np.nan
        else:

            with h5py.File(temp_path, 'r') as f:
                F_left_dots = np.array(f['raw/single_trial_dots_left'])
                F_right_dots = np.array(f['raw/single_trial_dots_right'])








            F0_left_dots = np.nanmean(F_left_dots[:, int(5 / dt):int(10 / dt)], axis=1, keepdims=True)
            df_F_left_dots = 100 * (F_left_dots - F0_left_dots) / F0_left_dots

            F0_right_dots = np.nanmean(F_right_dots[:, int(5 / dt):int(10 / dt)], axis=1, keepdims=True)
            df_F_right_dots = 100 * (F_right_dots - F0_right_dots) / F0_right_dots

            # Average over trials
            df_F_left_dots_avg = np.nanmean(df_F_left_dots, axis=0)
            df_F_right_dots_avg = np.nanmean(df_F_right_dots, axis=0)

            # Average F over trials
            df_F_left_dots_tavg = np.nanmean(df_F_left_dots[:, 20:100], axis=0)
            df_F_right_dots_tavg = np.nanmean(df_F_right_dots[:, 20:100], axis=0)
            # Std of F over trials
            df_F_left_dots_tstd = np.nanstd(df_F_left_dots[:, 20:100], axis=0)
            df_F_right_dots_tstd = np.nanstd(df_F_right_dots[:, 20:100], axis=0)
            # Reliability
            rel_left_dots=np.nanmean(df_F_left_dots_tavg/df_F_left_dots_tstd, axis=0)
            rel_right_dots=np.nanmean(df_F_right_dots_tavg/df_F_right_dots_tstd, axis=0)


        # Loop through all cells



            # As we have the cell registered to the z-brain, we know if it is on the left or right hemisphere
        if left_hemisphere:
            PD = df_F_left_dots_avg  # We drop the first and last 10 s, as this is how the regressors had been computed
            ND = df_F_right_dots_avg
        else:
            PD = df_F_right_dots_avg
            ND = df_F_left_dots_avg

        # Compute the correleation coefficient to all three regressors
        ci = [np.corrcoef(PD, regressors[0])[0, 1],
              np.corrcoef(PD, regressors[1])[0, 1],
              np.corrcoef(PD, regressors[2])[0, 1]]

        class_label = ['integrator','dynamic_threshold','motor_command'][np.argmax(ci)]



        prediction_string = f'regressor_predicted_class = "{class_label}"\n'
        correlation_test = f'correlation_test_passed = "{ci[np.argmax(ci)] > 0.80}"\n'

        meta = open(data_path / 'paGFP' / directory / f'{directory}metadata.txt', 'r')
        t = meta.read()
        if not t[-1:] == '\n':
            t = t + '\n'

        new_t = (t + prediction_string + correlation_test)
        meta.close()

        meta = open(data_path / 'paGFP' / directory / f'{directory}metadata_with_regressor.txt', 'w')
        meta.write(new_t)
        meta.close()

        temp_df = pd.DataFrame({'cell_name':[directory],
                                'manual_assigned_class':[eval(new_t.split('\n')[1][19:])[0]],
                                'predicted_class':[class_label],
                                'correlation':[ci[np.argmax(ci)]],
                                'passed_correclation':[ci[np.argmax(ci)] > 0.80],
                                'manual_matches_regressor':[eval(new_t.split('\n')[1][19:])[0]==class_label],
                                'rel_left':rel_left_dots,
                                'rel_right':rel_right_dots})
        if df is None:
            df = temp_df
        else:
            df = pd.concat([df, temp_df])
        manual_assigned = eval(new_t.split('\n')[1][19:])[0]
        plt.figure()
        plt.plot(PD/np.max(PD))
        plt.plot(regressors[np.argmax(ci)]/np.max(regressors[np.argmax(ci)]))
        plt.title(f'{directory}\nManual: {manual_assigned}\n Predicted: {class_label}')

        fig = plt.gcf()
        os.makedirs(data_path / 'make_figures_FK_output' / "regressors_on_paGFP",exist_ok=True)
        fig.savefig(data_path / 'make_figures_FK_output' / "regressors_on_paGFP" / (directory + '.png'))
    df = df.reset_index(drop=True)
