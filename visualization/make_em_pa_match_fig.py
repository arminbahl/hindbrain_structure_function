from hindbrain_structure_function.visualization.make_figures_FK_new import *

if __name__ == "__main__":
    # II
    kk = make_figures_FK(modalities=['pa', 'em'],
                         keywords='all',
                         use_smooth_pa=True,
                         mirror=True,
                         only_soma=True,
                         load_what='swc',
                         cell_type='')
    kk.all_cells = kk.all_cells.loc[kk.all_cells['cell_name'].isin(['133478', '20230628.2'])]
    kk.plot_y_projection(show_brs=True,
                         which_brs="raphe",
                         force_new_cell_list=False,
                         rasterize=False,
                         black_neuron=False,
                         standard_size=True,
                         volume_outlines=True,
                         background_gray=True,
                         only_soma=False,
                         midline=True,
                         plot_synapse_distribution=False)

    plt.show()
    # DT
    kk = make_figures_FK(modalities=['pa', 'em'],
                         keywords='all',
                         use_smooth_pa=True,
                         mirror=True,
                         only_soma=True,
                         load_what='swc',
                         cell_type='')
    kk.all_cells = kk.all_cells.loc[kk.all_cells['cell_name'].isin(['147009', '20230417.2'])]  # 141170 - 20230323.2
    kk.plot_y_projection(show_brs=True,
                         which_brs="raphe",
                         force_new_cell_list=False,
                         rasterize=False,
                         black_neuron=False,
                         standard_size=True,
                         volume_outlines=True,
                         background_gray=True,
                         only_soma=False,
                         midline=True,
                         plot_synapse_distribution=False)
    plt.show()

    kk = make_figures_FK(modalities=['pa', 'em'],
                         keywords='all',
                         use_smooth_pa=True,
                         mirror=True,
                         only_soma=True,
                         load_what='swc',
                         cell_type='')
    kk.all_cells = kk.all_cells.loc[kk.all_cells['cell_name'].isin(['121939', '20240219.2'])]  # 121939 - 20240219.2
    kk.plot_y_projection(show_brs=True,
                         which_brs="raphe",
                         force_new_cell_list=False,
                         rasterize=False,
                         black_neuron=False,
                         standard_size=True,
                         volume_outlines=True,
                         background_gray=True,
                         only_soma=False,
                         midline=True,
                         plot_synapse_distribution=False)
    plt.show()

    kk = make_figures_FK(modalities=['pa', 'em'],
                         keywords='all',
                         use_smooth_pa=True,
                         mirror=True,
                         only_soma=True,
                         load_what='swc',
                         cell_type='')
    kk.all_cells = kk.all_cells.loc[kk.all_cells['cell_name'].isin(['141963', '20230508.1'])]  # 141963 - 20230508.1
    kk.plot_y_projection(show_brs=True,
                         which_brs="raphe",
                         force_new_cell_list=False,
                         rasterize=False,
                         black_neuron=False,
                         standard_size=True,
                         volume_outlines=True,
                         background_gray=True,
                         only_soma=False,
                         midline=True,
                         plot_synapse_distribution=False)

    plt.show()
