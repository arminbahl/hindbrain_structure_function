import numpy as np

from analysis_helpers.analysis.personal_dirs.Florian.thesis_plot.Morphologies.utils.extract_traces_from_roi import *
from skimage import exposure


def trim_img(img,size_image,x,y,z):
    x_min, x_max = int(x - size_image / 2), int(x + size_image / 2)
    y_min, y_max = int(y - size_image / 2), int(y + size_image / 2)
    img = img[z,y_min:y_max,x_min:x_max,]
    return img

def draw_poly_around_center(fig,size_of_image,size_of_poly):
    fig.draw_polygon([(size_of_image/2)-(size_of_poly*size_of_image),
                    (size_of_image/2)-(size_of_poly*size_of_image),
                    (size_of_image/2)+(size_of_poly*size_of_image),
                    (size_of_image/2)+(size_of_poly*size_of_image),
                    (size_of_image/2)-(size_of_poly*size_of_image)],[(size_of_image/2)-(size_of_poly*size_of_image),
                                                             (size_of_image/2)+(size_of_poly*size_of_image),
                                                             (size_of_image/2)+(size_of_poly*size_of_image),
                                                             (size_of_image/2)-(size_of_poly*size_of_image),
                                                             (size_of_image/2)-(size_of_poly*size_of_image)],
                   lc="red",lw=0.5,alpha=0,line_dashes = (2,2))




#filename = '2023-03-21_11-04-42'
#df_f0_stim0, df_f0_stim1 = extract_traces_from_roi(filename)












def plot_single_trials_fig(experiment_name,
                           cell_register_row,
                           no_trials,
                           do_hcr,
                           gad1b_path=None,
                           vglut2a_path=None,
                           imaged950_800=True,
                           wavelength_950=None,
                           existing_plot=None,
                           no_y_plots=1,
                           current_y_plot=0,
                           custom_text="",
                           show_roi_on_stack=False):

    f = h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\functional\{experiment_name}\{experiment_name}_preprocessed_data.h5")
    stimulus_aligned_F0 = np.array(f["z_plane0000/manual_segmentation/stimulus_aligned_dynamics/stimulus0000/F"])
    stimulus_aligned_F1 = np.array(f["z_plane0000/manual_segmentation/stimulus_aligned_dynamics/stimulus0001/F"])
    f.close()
    if np.nanmin([np.nanmin(stimulus_aligned_F0), np.nanmin(stimulus_aligned_F1)])<0:
        stimulus_aligned_F0 = stimulus_aligned_F0 + np.abs(
            np.nanmin([np.nanmin(stimulus_aligned_F0), np.nanmin(stimulus_aligned_F1)]))
        stimulus_aligned_F1 = stimulus_aligned_F1 + np.abs(
            np.nanmin([np.nanmin(stimulus_aligned_F0), np.nanmin(stimulus_aligned_F1)]))

    df_f0_stim0 = (stimulus_aligned_F0[:, 0, :] - np.nanmean(stimulus_aligned_F0[:, 0, 0:20], axis=1)[:,
                                                  np.newaxis]) / np.nanmean(stimulus_aligned_F0[:, 0, 0:20], axis=1)[:,
                                                                 np.newaxis]
    df_f0_stim1 = (stimulus_aligned_F1[:, 0, :] - np.nanmean(stimulus_aligned_F1[:, 0, 0:20], axis=1)[:,
                                                  np.newaxis]) / np.nanmean(stimulus_aligned_F1[:, 0, 0:20], axis=1)[:,
                                                                 np.newaxis]

    average_dynamic0 = np.mean(df_f0_stim0, axis=0)
    average_dynamic1 = np.mean(df_f0_stim1, axis=0)

    if no_trials == 'all':
        no_trials = np.min([df_f0_stim0.shape[0],df_f0_stim1.shape[0]])
        
    x_pos_plots = [x*4+3 for x in range(no_trials+1)]
    x_pos_hcr = [np.max(x_pos_plots)+4,np.max(x_pos_plots)+6]

    my_dashes = [3,3]
    if existing_plot == None:
        fig = Figure(figure_title=f"",
                     fig_width= np.max(x_pos_plots)+15,
                     fig_height= 6.1*no_y_plots,
                     errorbar_area=False)
    else:
        fig = existing_plot



    text_plot = fig.create_plot(plot_label='',
                                         plot_title=f'',
                                         xpos=x_pos_plots[0],
                                         ypos=6.75+(current_y_plot*6),
                                         plot_height=3.5,
                                         plot_width=3.5,
                                         errorbar_area=False,
                                         xl="Time (s)",
                                         xmin=-5,
                                         xmax=70,
                                         xticks=[0, 10,30, 50, 60],
                                         yticks=[],yl="",
                                         legend_xpos=3.8,
                                         legend_ypos=24,
                                            axis_off = True
                                         )

    text_plot.draw_text(0,0,custom_text)
    list_of_plots = []

    max_stim0 = np.nanmax([np.nanmax(x) for x in df_f0_stim0[:no_trials,:]])*0.95
    max_stim1 = (np.nanmax([np.nanmax(x) for x in df_f0_stim1[:no_trials,:]]))*0.95
    #max_stim1 = [np.max(df_f0_stim0[x,:]) for x in range(len(df_f0_stim0.shape[0]))]
    y_upper_lim = round((np.nanmax([max_stim0,max_stim1])),2)
    list_of_plots.append(fig.create_plot(plot_label='a',
                                         plot_title=f'mean over\nall trials',
                                         xpos=x_pos_plots.pop(0),
                                         ypos=2+(current_y_plot*6),
                                         plot_height=3.5,
                                         plot_width=3.5,
                                         errorbar_area=False,
                                         xl="Time (s)",
                                         xmin=-5,
                                         xmax=70,
                                         xticks=[0, 10,30, 50, 60],
                                         yticks=[0,y_upper_lim],yl="ΔF/F",
                                         legend_xpos=3.8,
                                         legend_ypos=24,
                                         vspans=[[10,50,"#FF0000",0.1]]))

    list_of_plots[-1].draw_line(x=np.arange(0,60,0.5), y=average_dynamic0, lw=0.75,errorbar_area=False)
    list_of_plots[-1].draw_line(x=np.arange(0,60,0.5), y=average_dynamic1, lw=0.75,line_dashes = my_dashes,errorbar_area=False)



    for trial_number in range(no_trials):

        list_of_plots.append(fig.create_plot(plot_label="",
                                             plot_title=f'#{trial_number}',
                                             xpos=x_pos_plots.pop(0),
                                             ypos=2+(current_y_plot*6),
                                             plot_height=3.5,
                                             plot_width=3.5,
                                             errorbar_area=False,
                                             xl="Time (s)",
                                             xmin=-5,
                                             xmax=70,
                                             xticks=[0, 10,30, 50, 60],
                                             yticks=[0,y_upper_lim],
                                             yl="",
                                             legend_xpos=3.8,
                                             legend_ypos=24,
                                             vspans=[[10,50,"#FF0000",0.1]]))
        try:
            list_of_plots[-1].draw_line(x=np.arange(0,60,0.5), y=df_f0_stim0[trial_number,:], lw=0.75,errorbar_area=False)
        except:
            pass

        try:
            list_of_plots[-1].draw_line(x=np.arange(0, 60, 0.5), y=df_f0_stim1[trial_number, :], lw=0.75,
                                        line_dashes=my_dashes, errorbar_area=False)
        except:
            pass

        list_of_plots[-1].ax.sharey(list_of_plots[0].ax)



    if do_hcr:


        size_of_image = 50

        item["1020 nm with Gad1b"] = item["1020 nm with Gad1b"].rstrip()
        item["800 nm with Vglut2a"] = item["800 nm with Vglut2a"].rstrip()

        vglut_hcr_date = item["800 nm with Vglut2a"].split(' ')[-1]
        gad_hcr_date = item["1020 nm with Gad1b"].split(' ')[-1]

        if "_wv-800" in vglut_hcr_date or "-wv-800" in vglut_hcr_date:
            vglut_hcr_date = vglut_hcr_date[:-7]


        #vglut

        try:
            gcamp4vglut = np.array(h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\HCR\{vglut_hcr_date}\{vglut_hcr_date}_wv-800_preprocessed_data.h5")['average_stack_red_channel'])

        except:
            try:
                gcamp4vglut = np.array(h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\HCR\{vglut_hcr_date}\{vglut_hcr_date}-wv-800_preprocessed_data.h5")['average_stack_red_channel'])
            except:
                gcamp4vglut = np.array(h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\HCR\{vglut_hcr_date}\{vglut_hcr_date}_preprocessed_data.h5")['average_stack_red_channel'])

        try:
            vglut = np.array(h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\HCR\{vglut_hcr_date}\{vglut_hcr_date}_wv-800_preprocessed_data.h5")['average_stack_green_channel'])
        except:
            try:
                vglut = np.array(h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\HCR\{vglut_hcr_date}\{vglut_hcr_date}-wv-800_preprocessed_data.h5")['average_stack_green_channel'])
            except:
                vglut = np.array(h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\HCR\{vglut_hcr_date}\{vglut_hcr_date}_preprocessed_data.h5")['average_stack_green_channel'])





        locs_1020 = [int(x) for x in item['Location cell in HCR 1020'].split(',')]
        locs_800 = [int(x) for x in item['Location cell in HCR 800'].split(',')]

        vglut_loc_x,vglut_loc_y,vglut_z = locs_800[0],locs_800[1],locs_800[2]
        gcamp4vglut = trim_img(gcamp4vglut,size_of_image,vglut_loc_x,vglut_loc_y,vglut_z)
        vglut = trim_img(vglut,size_of_image,vglut_loc_x,vglut_loc_y,vglut_z)

        gcamp4vglut = np.clip(gcamp4vglut, np.percentile(gcamp4vglut, 1), np.percentile(gcamp4vglut, 99))
        vglut = np.clip(vglut, np.percentile(vglut, 1), np.percentile(vglut, 99))
        gcamp4vglut = exposure.adjust_gamma(gcamp4vglut, gamma=1)
        vglut = exposure.adjust_gamma(vglut, gamma=1)

        #gad1b

        gad = np.array(h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\HCR\{gad_hcr_date}\{gad_hcr_date}_preprocessed_data.h5")['average_stack_red_channel'])
        gcamp4gad = np.array(h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\HCR\{gad_hcr_date}\{gad_hcr_date}_preprocessed_data.h5")['average_stack_green_channel'])

        gad_loc_x,gad_loc_y,gad_z =  locs_1020[0],locs_1020[1],locs_1020[2]
        gcamp4gad = trim_img(gcamp4gad,size_of_image,gad_loc_x,gad_loc_y,gad_z)
        gad = trim_img(gad,size_of_image,gad_loc_x,gad_loc_y,gad_z)

        gcamp4gad = np.clip(gcamp4gad,np.percentile(gcamp4gad,1),np.percentile(gcamp4gad,99))
        gad = np.clip(gad,np.percentile(gad,1),np.percentile(gad,99))
        gcamp4gad = exposure.adjust_gamma(gcamp4gad, gamma=1)
        gad = exposure.adjust_gamma(gad, gamma=1)





        plot5 = fig.create_plot(plot_label='b', plot_title="Gad1b",
                                xpos=x_pos_hcr[0], ypos=4+(current_y_plot*6), plot_height=1.5, plot_width=1.5,
                                xmin=0, xmax=gad.shape[0],
                                ymin=gad.shape[1], ymax=0,
                                zl="Activity", zmin=np.min(gad), zmax=np.max(gad), zticks=[0, 128, 255], show_colormap=False)

        plot5.draw_image(img=gad, extent=None, image_interpolation='bilinear',aspect="equal",colormap="Reds")




        plot7 = fig.create_plot(plot_label='', plot_title="Vglut2a",
                                xpos=x_pos_hcr[1], ypos=4+(current_y_plot*6), plot_height=1.5, plot_width=1.5,
                                xmin=0, xmax=vglut.shape[0],
                                ymin=vglut.shape[1], ymax=0,
                                zl="Activity", zmin=np.min(vglut), zmax=np.max(vglut), zticks=[0, 128, 255], show_colormap=False)

        # plot6 = fig.create_plot(plot_label='', plot_title="GCaMP Gad1b",
        #                         xpos=np.mean(plot5.plot_dict['xpos']+plot7.plot_dict['xpos']), ypos=2+(current_y_plot*6), plot_height=1.5, plot_width=1.5,
        #                         xmin=0, xmax=gcamp4gad.shape[0],
        #                         ymin=gcamp4gad.shape[1], ymax=0,
        #                         zl="Activity", zmin=np.min(gcamp4gad), zmax=np.max(gcamp4gad), zticks=[0, 128, 255], show_colormap=False)

        plot6 = fig.create_plot(plot_label='', plot_title="GCaMP Gad1b",
                                xpos=np.mean(plot5.plot_dict['xpos']), ypos=2+(current_y_plot*6), plot_height=1.5, plot_width=1.5,
                                xmin=0, xmax=gcamp4gad.shape[0],
                                ymin=gcamp4gad.shape[1], ymax=0,
                                zl="Activity", zmin=np.min(gcamp4gad), zmax=np.max(gcamp4gad), zticks=[0, 128, 255], show_colormap=False)

        plot6.draw_image(img=gcamp4gad, extent=None, image_interpolation='bilinear',aspect="equal",colormap="gray")


        plot7.draw_image(img=vglut, extent=None, image_interpolation='bilinear',aspect="equal",colormap="Blues")



        plot8 = fig.create_plot(plot_label='', plot_title="GCaMP for Vglut2a",
                                xpos=plot7.plot_dict['xpos'], ypos=plot6.plot_dict['ypos'], plot_height=1.5, plot_width=1.5,
                                xmin=0, xmax=gcamp4vglut.shape[0],
                                ymin=gcamp4vglut.shape[1], ymax=0,
                                zl="Activity", zmin=np.min(gcamp4vglut), zmax=np.max(gcamp4vglut), zticks=[0, 128, 255], show_colormap=False)

        plot8.draw_image(img=gcamp4vglut, extent=None, image_interpolation='bilinear',aspect="equal",colormap="gray")


        size_of_poly = 0.15

        draw_poly_around_center(plot5,size_of_image,size_of_poly)
        draw_poly_around_center(plot6,size_of_image,size_of_poly)
        draw_poly_around_center(plot7,size_of_image,size_of_poly)
        #draw_poly_around_center(plot8,size_of_image,size_of_poly)

    if show_roi_on_stack:
        functional_average =np.array(h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\functional\{experiment_name}\{experiment_name}_preprocessed_data.h5")['average_stack_green_channel'])
        import tifffile
        print('image',functional_average.shape)
        try:
            roi  = tifffile.imread(rf"W:\Florian\function-neurotransmitter-morphology\functional\{experiment_name}\{experiment_name}_roi.tiff")
            print(roi.shape)
        except:

            roi = tifffile.imread(
                rf"W:\Florian\function-neurotransmitter-morphology\functional\{experiment_name}\{experiment_name}_roi.tif")
            print(roi.shape)
        plot = fig.create_plot(plot_label='', plot_title="ROI",
                                xpos=x_pos_hcr[0], ypos=2 + (current_y_plot * 6), plot_height=3.5, plot_width=3.5,
                                xmin=0, xmax=functional_average[0].shape[0],
                                ymin=0, ymax=functional_average[0].shape[1],
                                zl="Activity", zmin=np.min(functional_average[0]), zmax=np.max(functional_average[0]), zticks=[0, 128, 255],
                                show_colormap=False)

        plot.draw_image(img=functional_average[0], extent=None, image_interpolation='bilinear', aspect="equal", colormap="gray")
        plot.draw_image(img=roi, extent=None, image_interpolation='bilinear', aspect="equal", colormap="Reds",alpha=0.7)
        # plt.imshow(functional_average[0])
        # plt.imshow(roi,alpha=0.3,cmap="Greens_r")
        # plt.show()
    import datetime
    if no_y_plots-1 == current_y_plot:
        timestamp = datetime.datetime.now()
        os.makedirs(fr"C:\Users\ag-bahl\Desktop\plot_diary\{timestamp.strftime('%Y_%m_%d')}",exist_ok=True)
        fig.save(fr"C:\Users\ag-bahl\Desktop\plot_diary\{timestamp.strftime('%Y_%m_%d')}\{timestamp.strftime('%Y_%m_%d_%H_%M_%S_%f')}.{'pdf'}", open_file=True)
    return fig

import pandas as pd
cell_register = pd.read_csv(r"C:\Users\ag-bahl\Downloads\Table1.csv")
cell_register[cell_register['Traced'].isna()==False]
do_hcr=True
if do_hcr:
    cell_register = cell_register.loc[(cell_register["Location cell in HCR 1020"].isna() == False) & (
                cell_register["Location cell in HCR 800"].isna() == False), :]

cell_register= cell_register.loc[cell_register['ROI']=="Y",:]
cell_register['Dynamic Type'] = cell_register['Dynamic Type'].apply(lambda x: x.lower())
cell_register = cell_register.sort_values(["Dynamic Type",'Function','Manually evaluated cell type'],ascending=True)
cell_register = cell_register.reset_index(drop=True)

for i,item in cell_register.iterrows():
    temp_function = item["Function"].split(' ')[-1]
    temp_cell_type = item["Manually evaluated cell type"]
    if f'{temp_function}_preprocessed_data.h5' in os.listdir(rf"W:\Florian\function-neurotransmitter-morphology\functional\{temp_function}"):
        f = h5py.File(rf"W:\Florian\function-neurotransmitter-morphology\functional\{temp_function}\{temp_function}_preprocessed_data.h5")
    else:
        continue
    print('name:',item["Internal name"],'gad:',item['1020 nm with Gad1b'].rstrip().split(' ')[-1],"vglut:",item['800 nm with Vglut2a'].rstrip().split(' ')[-1])
    if 'manual_segmentation' in f['z_plane0000']:
        f.close()
        #print(np.argwhere(cell_register.index == item.name)[0][0])
        if i == cell_register.index[0]:

            temp_fig = plot_single_trials_fig(temp_function,cell_register_row=item,
                                              no_trials = 5,#39,
                                              do_hcr=do_hcr,
                                              no_y_plots=cell_register.shape[0],
                                              custom_text=f'{item["Manually evaluated cell type"]} {temp_cell_type}',show_roi_on_stack=False)
        else:
            plot_single_trials_fig(temp_function,cell_register_row=item,
                                   no_trials = 5,#39,
                                   do_hcr=do_hcr,
                                   existing_plot=temp_fig,
                                   no_y_plots=cell_register.shape[0],
                                   current_y_plot=np.argwhere(cell_register.index == item.name)[0][0],
                                   custom_text=f'{item["Internal name"]} {temp_cell_type}',show_roi_on_stack=False)
    else:
        print(temp_function)
        f.close()



cell_register_all = pd.read_csv(r"C:\Users\ag-bahl\Downloads\Table1.csv")
cell_register_all = cell_register_all[cell_register_all['Function'].isna()==False]
cell_register_all = cell_register_all.sort_values('Manually evaluated cell type')

plot_single_trials_fig('2023-06-28_16-43-13','all',do_hcr=False,no_trials = 10,)
