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
    ci.all_cells = ci.all_cells.loc[ci.all_cells['cell_name'].isin(['133478', '20230628.2'])]
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

    # MC 20230613.1
    for em_cell in ['168586', '153284', '147321', '166323', '124806', '126984']:
        mc1 = make_figures_FK(modalities=['pa', 'em'],
                              keywords='all',
                              use_smooth_pa=True,
                              mirror=True,
                              only_soma=True,
                              load_what='swc',
                              cell_type='')
        mc1.all_cells = mc1.all_cells.loc[
            mc1.all_cells['cell_name'].isin([em_cell, '20230613.1'])]  # 141963 - 20230508.1
        mc1.plot_y_projection(show_brs=True,
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

    # MC cell 20230527.1
    for em_cell in ['66813', '125979']:
        mc2 = make_figures_FK(modalities=['pa', 'em'],
                              keywords='all',
                              use_smooth_pa=True,
                              mirror=True,
                              only_soma=True,
                              load_what='swc',
                              cell_type='')
        mc2.all_cells = mc2.all_cells.loc[
            mc2.all_cells['cell_name'].isin([em_cell, '20230527.1'])]  # 141963 - 20230508.1
        mc2.plot_y_projection(show_brs=True,
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
