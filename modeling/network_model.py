from numba import jit
import numpy as np
import pylab as pl
import pathlib

# matplotlib.use("macosx")
import matplotlib.colors as colors
import numpy as np
import scipy

from analysis_helpers.analysis.utils.figure_helper import Figure


dt = 0.01
t = np.arange(0, 60, dt)

#names = np.array(["IIel", "Xl",  "CIl",  "DTl",  "Ml",  "IIer", "Xr",  "CIr",  "DTr",  "Mr"])
names = np.array(["PI+_l", "iMI+_l",  "iMI-_l", "CMI-_l", "MON-_l", "SMI_l", "MY_l",
                  "PI+_r", "iMI+_r",  "iMI-_r", "CMI-_r", "MON-_r", "SMI_r", "MY__r"])

# Buffers for all cell types firing rates on the left and right
rates1 = np.zeros((len(t), 14))
rates2 = np.zeros((len(t), 14))
S_left1 = np.zeros(len(t))
S_right1 = np.zeros(len(t))
S_left2 = np.zeros(len(t))
S_right2 = np.zeros(len(t))

# # 7 x 14 matri
# connectivity_weights = [
#     [0,  1,  0,  0,  1,  0,  0,     0,  0,  0,  0,  0,  0,  0], # eVI_l
#     [0,  1,  1,  1,  1,  1,  0,     0,  0,  0,  0,  0,  0,  0], # eII_l
#     [0,  0,  0,  0, -5.5,  0,  0,     0,  0,  0,  0,  0,  0,  0], # iII_l
#     [0,  0,  0,  0,  0,  0,  0,     0, -1,  0, -1, -1, -1,  0], # iCI_l
#     [0, -1,  0, -1, -1,  -1,  0,     0,  0,  0, -1,  0,  0,  0], # iDT_l
#     [0,  0,  0,  0,  0,  0,  1,     0,  0,  0,  0,  0,  0,  1], # eMC_l
#     [0,  0,  0,  0,  0,  0,  0,     0,  0,  0,  0,  0,  0,  0]] # My_l

# Old model matrix from 2020 paper
# 7 x 14 matri
connectivity_weights = [
    [0,  1,  0,  0,  1,  0,  0,     0,  0,  0,  0,  0,  0,  0], # eVI_l
    [0,  1,  0,  1,  0,  1,  0,     0,  0,  0,  0,  0,  0,  0], # eII_l
    [0,  0,  0,  0, 0,  0,  0,     0,  0,  0,  0,  0,  0,  0], # iII_l
    [0,  0,  0,  0,  -3.0,  0,  0,     0, -1,  0, 0, 0, 0,  0], # iCI_l
    [0, 0,  0, 0, 0,  -1,  0,     0,  0,  0, 0,  0,  0,  0], # iDT_l
    [0,  0,  0,  0,  0,  0,  0,     0,  0,  0,  0,  0,  0,  0], # eMC_l
    [0,  0,  0,  0,  0,  0,  0,     0,  0,  0,  0,  0,  0,  0]] # My_l


# Leaks lead to forgetting (fraction per time bin), they define the time constant of the cell without recurrent feedbacks
leaks = np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]) * 0 + 2.5


#
# # old model
# connectivity_weights = [
#     [0,  1,  0,  0,  1,  1,  0,     0,  0,  0,  0,  0,  0,  0], # eVI_l
#     [0,  1,  1,  1,  1,  1,  0,     0,  0,  0,  0,  0,  0,  0], # eII_l
#     [0,  0,  0,  0, -5.5,  0,  0,     0,  0,  0,  0,  0,  0,  0], # iII_l
#     [0,  0,  0,  0,  0,  0,  0,     0, -1,  0, -1, -1, -1,  0], # iCI_l
#     [0, -1,  0, -1, -1,  -1,  0,     0,  0,  0, -1,  0,  0,  0], # iDT_l
#     [0,  0,  0,  0,  0,  0,  1,     0,  0,  0,  0,  0,  0,  1], # eMC_l
#     [0,  0,  0,  0,  0,  0,  0,     0,  0,  0,  0,  0,  0,  0]] # My_l


# Both visual inputs have some baseline input rate
S_left1[:] = 0.5
S_right1[:] = 0.5

# Show some input on the left eye
S_left1[int(20/dt):int(40/dt)] += 1
# Which reduced firing on the right side.
S_right1[int(20/dt):int(40/dt)] -= 0.5

S_left2[:] = 0.5
S_right2[:] = 0.5

# Show some input on the left eye
# S_left2[int(20/dt):int(40/dt)] += np.linspace(0, 1, int(20/dt))
# S_right2[int(20/dt):int(40/dt)] -= 0.5*S_left2[int(20/dt):int(40/dt)]

# Opposing stimulation
S_left2[int(20/dt):int(30/dt)] += 1
S_right2[int(20/dt):int(30/dt)] -= 0.5
S_left2[int(30/dt):int(40/dt)] -= 0.5
S_right2[int(30/dt):int(40/dt)] += 1

#S_left2[int(20/dt):int(40/dt)] += 0.5*(-np.cos(2*np.pi*0.05 * (t-20))[int(20/dt):int(40/dt)] + 1)
#S_right2[int(20/dt):int(40/dt)] -= 0.5*S_left2[int(20/dt):int(40/dt)]

# Show some input on the left eye
#S_left2[int(20/dt):int(40/dt)] -= 0.5*np.linspace(0, 1, int(20/dt))
# Which reduced firing on the right side.
#S_right2[int(30/dt):int(40/dt)] += 1

#@jit(nopython=True)
def run_simulation(t, S_left, S_right, rates):

    for i_t in range(1, len(t)):

        # Loop over all columns and compute their input
        for i_cell in range(14):

            # Each cell has it's own leak, representing generell cell biophysics
            d_rate = -leaks[i_cell % 7] * rates[i_t-1, i_cell]
            if i_t == 1:
                print(f"d {names[i_cell]} = {-leaks[i_cell % 7]:.1f} * {names[i_cell]}", end='')

                if i_cell == 0:
                    print(" + Stim_l(t)", end='')
                elif i_cell == 7:
                    print(" + Stim_r(t)", end='')

            if i_cell == 0:
                d_rate += S_left[i_t - 1]
            if i_cell == 7:
                d_rate += S_right[i_t - 1]

            if i_cell < 7:
                # Loop over all ipsilateral rows
                for j_cell in range(7):
                    w = connectivity_weights[j_cell][i_cell]

                    if abs(w) > 0:
                        d_rate += w * rates[i_t-1, j_cell]
                        if i_t == 1:
                            print(f" + {w:.1f} * {names[j_cell]}", end='')

                # Loop over all contraleral rows
                for j_cell in range(7):
                    w = connectivity_weights[j_cell][i_cell + 7]
                    if abs(w) > 0:
                        d_rate += w * rates[i_t-1, j_cell + 7]
                        if i_t == 1:
                            print(f" + {w:.1f} * {names[j_cell + 7]}", end='')

            else:
                # Loop over all ipsilateral rows
                for j_cell in range(7):
                    w = connectivity_weights[j_cell][i_cell-7]
                    if abs(w) > 0:
                        d_rate += w * rates[i_t-1, j_cell + 7]
                        if i_t == 1:
                            print(f" + {w:.1f} * {names[j_cell + 7]}", end='')

                # Loop over all contraleral rows
                for j_cell in range(7):
                    w = connectivity_weights[j_cell][i_cell]
                    if abs(w) > 0:
                        d_rate += w * rates[i_t-1, j_cell]
                        if i_t == 1:
                            print(f" + {w:.1f} * {names[j_cell]}", end='')

            rates[i_t, i_cell] = rates[i_t-1, i_cell] + d_rate*dt

            # #Ablate the dt!
            # if i_cell == 4 or i_cell == 4+7:
            #     rates[i_t, i_cell] = 0

            if rates[i_t, i_cell] < 0.00:
                 rates[i_t, i_cell] = 0.00

            # if i_cell == 4 or i_cell == 4+7:
            #     rates[i_t, i_cell] += 0.0005

            if i_t == 1:
                print('')

run_simulation(t, S_left1, S_right1, rates1)
run_simulation(t, S_left2, S_right2, rates2)

# Do a time constant fit (average time to 90% as in

# Make a standard figure
fig = Figure(figure_title="Figure 5")

colors = ["#00FF00", "#FEB326", "#FEB326", "#E84D8A", "#26A8DF", "#7F58AF", "#5CE572"]
line_dashes = [None, None, (2,2), None, None, None, None]

for prediction in [0, 1]:


    for ipsi in [0, 1]:
        plot0 = fig.create_plot(plot_label='a', xpos=3.5 + ipsi * 4 + prediction * 8, ypos=24.5, plot_height=0.5, plot_width=2,
                                errorbar_area=True,
                                xl="", xmin=9, xmax=51, xticks=[],
                                plot_title="Ipsi. hemisphere" if ipsi == 0 else "Contra. hemisphere",
                                yl="", ymin=-0.1, ymax=1.6, yticks=[0, 0.5], hlines=[0],
                                vspans=[[20, 40, "#aaaaaa", 0.5]])

        plot1 = fig.create_plot(plot_label='a', xpos=3.5 + ipsi * 4 + prediction * 8, ypos=22, plot_height=2, plot_width=2, errorbar_area=True,
                                xl="Time (s)", xmin=9, xmax=51, xticks=[10, 20],
                                yl="Δr / r0", ymin=-2, ymax=7.1, yticks=[0, 3.5, 7], hlines=[0],
                                vspans=[[20, 40, "#aaaaaa", 0.5]])

        if prediction == 0 and ipsi == 0:
            plot0.draw_line(x=t[int(10 / dt):-int(10 / dt)], y=S_left1[int(10 / dt):-int(10 / dt)], lw=1.5, lc='black')
        if prediction == 0 and ipsi == 1:
            plot0.draw_line(x=t[int(10 / dt):-int(10 / dt)], y=S_right1[int(10 / dt):-int(10 / dt)], lw=1.5, lc='black')

        if prediction == 1 and ipsi == 0:
            plot0.draw_line(x=t[int(10 / dt):-int(10 / dt)], y=S_left2[int(10 / dt):-int(10 / dt)], lw=1.5, lc='black')
        if prediction == 1 and ipsi == 1:
            plot0.draw_line(x=t[int(10 / dt):-int(10 / dt)], y=S_right2[int(10 / dt):-int(10 / dt)], lw=1.5, lc='black')

        if prediction == 0:
            r0 = rates1[int(10/dt):int(20/dt)].mean(axis=0, keepdims=True)
            dr_ro = (rates1 - r0) / r0
        else:
            r0 = rates2[int(10 / dt):int(20 / dt)].mean(axis=0, keepdims=True)
            dr_ro = (rates2 - r0) / r0

        delays = []
        for i_cell in np.arange(7):
            rate_90_percent = dr_ro[int(20/dt):int(40/dt), i_cell + ipsi*7].max() * 0.9

            ind = np.where(dr_ro[int(20/dt):int(40/dt), i_cell + ipsi*7] > rate_90_percent)
            if len(ind[0]) > 0:
                t_to_90_percent = ind[0][0] * dt
                print("time to 90%", names[i_cell + ipsi*7], t_to_90_percent)

                delays.append(t_to_90_percent)
            else:
                delays.append(np.nan)

        delays2 = []
        for i_cell in np.arange(7):
            if prediction == 0:
                val_10_percent = dr_ro[int(40 / dt), i_cell + ipsi * 7] * 0.1
                ind = np.where(dr_ro[int(40 / dt):, i_cell + ipsi * 7] < val_10_percent)
            else:
                val_10_percent = dr_ro[int(30/dt), i_cell + ipsi*7]*0.1
                ind = np.where(dr_ro[int(30/dt):, i_cell + ipsi*7] < val_10_percent)

            if len(ind[0]) > 0:
                t_to_10_percent = ind[0][0] * dt
                print("time to 10%", names[i_cell + ipsi * 7], t_to_10_percent)
                delays2.append(t_to_10_percent)
            else:
                delays2.append(np.nan)

        for i in np.arange(0, 7):
            plot1.draw_line(x=t[int(10/dt):-int(10/dt)], y=dr_ro[int(10/dt):-int(10/dt), i + ipsi*7], lw=1.5, lc=colors[i], line_dashes=line_dashes[i], label=f"{names[i]}" if ipsi == 0 else None)

        # ######################
        # # Draw vertical bars
        # plot2 = fig.create_plot(plot_label='c', xpos=3.5 + ipsi * 4 + prediction * 8, ypos=20, plot_height=1.25, plot_width=1.5,
        #                         xl="Cell type", xmin=-0.5, xmax=6.5, xticks=[0, 1, 2, 3, 4, 5, 6], xticklabels=names[:7], xticklabels_rotation=45,
        #                         hlines=[0],
        #                         yl="Rise time (s)", ymin=-0.1, ymax=6.1, yticks=[0, 3, 6])
        #
        #
        # for i in range(7):
        #     plot2.draw_vertical_bars(x=[i], y=[delays[i]], lc=colors[i], vertical_bar_width=0.75)
        #
        # plot2 = fig.create_plot(plot_label='c', xpos=3.5 + ipsi * 4 + prediction * 8, ypos=17, plot_height=1.25,
        #                         plot_width=1.5,
        #                         xl="Cell type", xmin=-0.5, xmax=6.5, xticks=[0, 1, 2, 3, 4, 5, 6],
        #                         xticklabels=names[:7], xticklabels_rotation=45,
        #                         hlines=[0],
        #                         yl="Decay time (s)", ymin=-0.1, ymax=6.1, yticks=[0, 3, 6])
        #
        # for i in range(7):
        #     plot2.draw_vertical_bars(x=[i], y=[delays2[i]], lc=colors[i], vertical_bar_width=0.75)
        #

        plot2 = fig.create_plot(plot_label='c', xpos=3.5 + ipsi * 4 + prediction * 8, ypos=14, plot_height=1.25,
                                plot_width=1.5,
                                xl="Cell type", xmin=-0.5, xmax=6.5, xticks=[0, 1, 2, 3, 4, 5, 6],
                                xticklabels=names[:7], xticklabels_rotation=45,
                                hlines=[0, 1],
                                yl="Rise/Decay time ratio", ymin=-0.1, ymax=3.6, yticks=[0, 1.0, 2.0, 3.0])

        for i in range(7):
            plot2.draw_scatter(x=[i], y=[delays[i]/delays2[i]], pc=colors[i], alpha=1.0, ps=3.0, ec=colors[i])


        plot2 = fig.create_plot(plot_label='a', xpos=3.5 + ipsi * 4 + prediction * 8, ypos=9.5, plot_height=1.5,
                                plot_width=1.5,
                                errorbar_area=True,
                                xl="Rise time (s)", xmin=-0.1, xmax=6.1, xticks=[0, 3, 6],
                                yl="Decay time (s)", ymin=-0.1, ymax=6.1, yticks=[0, 3, 6], hlines=[0], vlines=[0])
        plot2.draw_line([0, 6], [0,6], lw=0.5, line_dashes=(2,2), lc='gray')
        for i in range(7):
            plot2.draw_scatter(x=[delays[i]], y=[delays2[i]], pc=colors[i], alpha=1.0, ps=3.0, ec=colors[i])


for prediction in [0, 1]:

    # Stats plot for the experimental data
    plot2 = fig.create_plot(plot_label='c', xpos=3.5 + prediction * 8, ypos=4, plot_height=1.25,
                                plot_width=1.5,
                                xl="Cell type", xmin=-0.5, xmax=6.5, xticks=[0, 1, 2],
                                xticklabels=["MI", "MON", "SMI"], xticklabels_rotation=45,
                                hlines=[0, 1],
                                yl="Rise/Decay time ratio", ymin=-0.1, ymax=2.4, yticks=[0, 0.5, 1.0, 1.5, 2.0])

    if prediction == 0:
        rise_times = [[ 4.,  12.5,  np.nan],
                         [ 3.5, 13.,  22.5],
                         [ 3.5, 10.5,  np.nan],
                         [ 4.,  12.5, 22.5],
                         [ 4.,  10.,  29.5],
                         [ 3.5, 12.5, 21.5],
                         [ np.nan, 22.5, 24. ]]

        decay_times = [[ 3.,  31.,   np.nan],
                     [ 2.5, 19.5, 27. ],
                     [ 2.5, 24.5,  np.nan],
                     [ 4.,  29.,  23. ],
                     [ 2.5, 27.5, 17. ],
                     [ 3.,   np.nan, 24. ],
                     [ np.nan,  np.nan, 34. ]]
    else:
        rise_times = [[4.0000, 10.5000, np.nan],
                      [3.5000, 10.0000, 13.5000],
                      [3.0000, 7.5000, np.nan],
                      [4.0000, 9.0000, 13.5000],
                      [6.5000, 12.0000, 14.0000],
                      [3.5000, 10.0000, 12.0000],
                      [np.nan, 7.5000, 3.5000]]

        decay_times = [[2.0000, 5.0000, np.nan],
                       [2.5000, 5.5000, 19.5000],
                       [2.0000, 4.5000, np.nan],
                       [2.5000, 6.0000, 6.5000],
                       [3.0000, 6.0000, 6.5000],
                       [2.0000, 7.0000, 4.0000],
                       [np.nan, 5.0000, 4.5000]]

    rise_times = np.array(rise_times)
    decay_times = np.array(decay_times)

    # MON, MI, SMI
    colors = ["#26A8DF", "#ED7658", "#7F58AF"]
    for j in range(3):
        j_ = [1,0,2][j] # change the order of the cells to MI, MON, SMI

        ratios = []
        for i in range(7):
            ratios.append(rise_times[i][j_]/decay_times[i][j_])
            plot2.draw_scatter(x=[j], y=[rise_times[i][j_]/decay_times[i][j_]], pc=colors[j_], alpha=0.33, ps=2.0, ec=colors[j_])

        mean = np.nanmean(ratios)
        sem = scipy.stats.sem(ratios, nan_policy='omit')

        plot2.draw_scatter(x=[j], y=[mean], yerr=[sem], pc=colors[j_], alpha=1.0, ps=3.0, ec=colors[j_])

    plot2 = fig.create_plot(plot_label='a', xpos=3.5 + prediction * 8, ypos=6.5, plot_height=1.5,
                            plot_width=1.5,
                            errorbar_area=True,
                            xl="Rise time (s)", xmin=-0.1, xmax=36.1, xticks=[0, 15, 30],
                            yl="Decay time (s)", ymin=-0.1, ymax=36.1, yticks=[0, 15, 30], hlines=[0], vlines=[0])
    plot2.draw_line([0, 36], [0, 36], lw=0.5, line_dashes=(2, 2), lc='gray')

    for j in range(3):
        j_ = [1, 0, 2][j]  # change the order of the cells to MI, MON, SMI
        for i in range(7):
            plot2.draw_scatter(x=[rise_times[i][j_]], y=[decay_times[i][j_]], pc=colors[j_], alpha=0.33, ps=2.0, ec=colors[j_])

        x_mean = np.nanmean(rise_times[:, j_])
        y_mean = np.nanmean(decay_times[:, j_])

        x_sem = scipy.stats.sem(rise_times[:, j_], nan_policy='omit')
        y_sem = scipy.stats.sem(decay_times[:, j_], nan_policy='omit')


        plot2.draw_scatter(x=[x_mean], y=[y_mean], pc=colors[j_], xerr=[x_sem], yerr=[y_sem], alpha=1.0, ps=3.0, ec=colors[j_])


fig.save(pathlib.Path.home() / 'Desktop' / "fig_test.pdf", open_file=True)






