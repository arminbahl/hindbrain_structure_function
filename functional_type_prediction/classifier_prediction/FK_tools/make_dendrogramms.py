import os
import matplotlib.pyplot as plt
import navis

def make_dendrogramms(path,cells):
    # dendrogramm
    os.makedirs(path.joinpath("make_figures_FK_output").joinpath("dendrogramms"), exist_ok=True)

    color_cell_type_dict_dend = {
        "integrator": (255, 99, 71, 1),
        "dynamic threshold": (100, 197, 235, 1),
        "motor command": (127, 88, 175, 0.7),
        "dynamic_threshold": (100, 197, 235, 1),
        "motor_command": (127, 88, 175, 1),
        'nan': (60, 60, 60, 1)
    }
    for i, cell in cells.iterrows():
        fig = plt.figure()
        navis.plot_flat(cell['swc'], layout='subway', plot_connectors=True, color='#{:02x}{:02x}{:02x}'.format(*color_cell_type_dict_dend[cell['function']]))
        plt.title(cell['imaging_modality'] + " " + str(cell['cell_name']))
        plt.axis('off')
        plt.savefig(path.joinpath("make_figures_FK_output").joinpath('dendrogramms').joinpath(rf"{cell['imaging_modality']}_{str(cell['cell_name'])}.jpg"), dpi=400)
        plt.close('all')
