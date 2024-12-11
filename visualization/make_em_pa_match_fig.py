from hindbrain_structure_function.visualization.make_figures_FK_new import *

if __name__ == "__main__":
    # CI
    ci = make_figures_FK(modalities=['pa', 'em'],
                         keywords='all',
                         use_smooth_pa=True,
                         mirror=True,
                         only_soma=True,
                         load_what='swc',
                         cell_type='')
    ci.all_cells = ci.all_cells.loc[ci.all_cells['cell_name'].isin(['131334', '20230428.1'])]
    ci.plot_y_projection(show_brs=True,
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
    dt = make_figures_FK(modalities=['pa', 'em'],
                         keywords='all',
                         use_smooth_pa=True,
                         mirror=True,
                         only_soma=True,
                         load_what='swc',
                         cell_type='')
    dt.all_cells = dt.all_cells.loc[dt.all_cells['cell_name'].isin(['147009', '20230417.2'])]  # 141170 - 20230323.2
    dt.plot_y_projection(show_brs=True,
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
    # II
    ii = make_figures_FK(modalities=['pa', 'em'],
                         keywords='all',
                         use_smooth_pa=True,
                         mirror=True,
                         only_soma=True,
                         load_what='swc',
                         cell_type='')
    ii.all_cells = ii.all_cells.loc[ii.all_cells['cell_name'].isin(['121939', '20240219.2'])]  # 121939 - 20240219.2
    ii.plot_y_projection(show_brs=True,
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

    # MC 

    mc = make_figures_FK(modalities=['pa', 'em'],
                         keywords='all',
                         use_smooth_pa=True,
                         mirror=True,
                         only_soma=True,
                         load_what='swc',
                         cell_type='')
    mc.all_cells = mc.all_cells.loc[
        mc.all_cells['cell_name'].isin(['66813', '20230527.1'])]  # 141963 - 20230508.1
    mc.plot_y_projection(show_brs=True,
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
