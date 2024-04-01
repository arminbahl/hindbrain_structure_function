import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import navis
from pathlib import Path
import pandas as pd
from hindbrain_structure_function.visualization.FK_tools.load_pa_table import *
from hindbrain_structure_function.visualization.FK_tools.load_clem_table import *
from hindbrain_structure_function.visualization.FK_tools.load_mesh import *
from hindbrain_structure_function.visualization.FK_tools.load_brs import *
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
from datetime import datetime
import plotly
import matplotlib
matplotlib.use('TkAgg')
class make_figures_FK:
    def __init__(self,
                 modalities = ['pa'],
                 keywords = ['integrator','contralateral']):
        
        #set_name_time
        self.name_time = datetime.now()
        
        #path settings
        self.path_to_data =  get_base_path() #path to clone of nextcloud, set your path in path_configuration.txt


        #load pa  table
        if 'pa' in modalities:
            pa_table = load_pa_table(self.path_to_data.joinpath("paGFP").joinpath("photoactivation_cells_table.csv"))
        #load clem table
        if 'clem' in modalities:
            clem_table = load_clem_table(self.path_to_data.joinpath('clem_zfish1').joinpath('all_cells'))

        #TODO here the loading of gregor has to go


        #concat tables
        if len(modalities)>1:
            all_cells = pd.concat([eval(x+'_table') for x in modalities])
        elif len(modalities) ==1:
            all_cells = eval(modalities[0]+"_table")
        all_cells = all_cells.reset_index(drop=True)

        #subset dataset for keywords
        for keyword in keywords:
            subset_for_keyword = all_cells['cell_type_labels'].apply(lambda current_label: True if keyword.replace("_"," ") in current_label else False)
            all_cells = all_cells[subset_for_keyword]



        all_cells['soma_mesh'] = np.nan
        all_cells['dendrite_mesh'] = np.nan
        all_cells['axon_mesh'] = np.nan
        all_cells['neurites_mesh'] = np.nan

        all_cells['soma_mesh'] = all_cells['soma_mesh'].astype(object)
        all_cells['dendrite_mesh'] = all_cells['dendrite_mesh'].astype(object)
        all_cells['axon_mesh'] = all_cells['axon_mesh'].astype(object)
        all_cells['neurites_mesh'] = all_cells['neurites_mesh'].astype(object)

        #set imaging modality to clem if jon scored it
        all_cells.loc[all_cells['tracer_names']=='Jonathan Boulanger-Weill','imaging_modality'] = 'clem' #TODO ask jonathan if we can write clem as imaging modality

        #load the meshes for each cell that fits queries in selected modalities
        for i,cell in all_cells.iterrows():
            all_cells.loc[i,:] = load_mesh(cell,self.path_to_data)

        self.all_cells = all_cells
    def plot_z_projection(self,show_brs = False,force_new_cell_list=False,xlim=[-700, -200]):
        #define colors for cells
        color_cell_type_dict = {"integrator":"red",
                                "dynamic threshold":"blue",
                                "motor command":"purple",}

        #load needed meshes into a list and assign colors

        if not "visualized_cells" in self.__dir__() or force_new_cell_list:
            self.visualized_cells = []
            self.color_cells = []
            for i,cell in self.all_cells.iterrows():

                       for label in cell.cell_type_labels:
                           if label.replace("_"," ") in color_cell_type_dict.keys():
                               temp_color = color_cell_type_dict[label.replace("_"," ")]
                               break
                       for key in ["soma_mesh", "axon_mesh", "dendrite_mesh", "neurites_mesh"]:
                            if not type(cell[key]) == float:
                               self.visualized_cells.append(cell[key])
                               if key != "dendrite_mesh":
                                   self.color_cells.append(temp_color)
                               elif key == "dendrite_mesh":
                                   self.color_cells.append("black")



        #here we start the plotting
        if show_brs:
            brain_meshes = load_brs(self.path_to_data,load_FK_regions=True)
            selected_meshes = ["Retina", 'Midbrain', "Forebrain", "Habenula", "Hindbrain", "Spinal Cord"]
            brain_meshes = [mesh for mesh in brain_meshes if mesh.name in selected_meshes]
            color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)

        #zprojection
        if show_brs:
            fig, ax = navis.plot2d(brain_meshes,color=color_meshes,alpha=0.2,linewidth=0.5,method='2d',view=('x', "-y"),group_neurons=True,rasterize=True)
            fig, ax = navis.plot2d(self.visualized_cells,color=self.color_cells,alpha=1,linewidth=0.5,method='2d',view=('x', "-y"),group_neurons=True,rasterize=True,ax=ax)
        else:
            fig, ax = navis.plot2d(self.visualized_cells, color=self.color_cells, alpha=1, linewidth=0.5, method='2d', view=('x', "-y"),
                                   group_neurons=True, rasterize=True)
        plt.xlim(xlim)
        if show_brs:
            brkw = "_with_brs_"
        else:
            brkw = "_without_brs_"

        #ax.set_ylim(-700, -200)
        os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("pdf"),exist_ok=True)
        os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("png"),exist_ok=True)
        os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("svg"),exist_ok=True)
        plt.savefig(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("pdf").joinpath(rf"z_projection{brkw}{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=1200)
        print("PDF saved!")
        plt.savefig(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("png").joinpath(rf"z_projection{brkw}{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"), dpi=1200)
        print("PNG saved!")
        plt.savefig(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("svg").joinpath(rf"z_projection{brkw}{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=1200)
        print("SVG saved!")

    def plot_y_projection(self, show_brs=False, force_new_cell_list=False):
        # define colors for cells
        color_cell_type_dict = {"integrator": "red",
                                "dynamic threshold": "blue",
                                "motor command": "purple", }

        # load needed meshes into a list and assign colors

        if not "visualized_cells" in self.__dir__() or force_new_cell_list:
            self.visualized_cells = []
            self.color_cells = []
            for i, cell in self.all_cells.iterrows():

                for label in cell.cell_type_labels:
                    if label.replace("_", " ") in color_cell_type_dict.keys():
                        temp_color = color_cell_type_dict[label.replace("_", " ")]
                        break
                for key in ["soma_mesh", "axon_mesh", "dendrite_mesh", "neurites_mesh"]:
                    if not type(cell[key]) == float:
                        self.visualized_cells.append(cell[key])
                        if key != "dendrite_mesh":
                            self.color_cells.append(temp_color)
                        elif key == "dendrite_mesh":
                            self.color_cells.append("black")

        # here we start the plotting
        if show_brs:
            brkw = "_with_brs_"
        else:
            brkw = "_without_brs_"


        if show_brs:
            brain_meshes = load_brs(self.path_to_data, load_FK_regions=True)
            selected_meshes = ["Retina", 'Midbrain', "Forebrain", "Habenula", "Hindbrain", "Spinal Cord"]
            brain_meshes = [mesh for mesh in brain_meshes if mesh.name in selected_meshes]
            color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)


        # zprojection
        if show_brs:
            fig, ax = navis.plot2d(brain_meshes, color=color_meshes, alpha=0.2, linewidth=0.5, method='2d',
                                   view=('x', "z"), group_neurons=True, rasterize=True)
            fig, ax = navis.plot2d(self.visualized_cells, color=self.color_cells, alpha=1, linewidth=0.5, method='2d',
                                   view=('x', "z"), group_neurons=True, rasterize=True, ax=ax)
        else:
            fig, ax = navis.plot2d(self.visualized_cells, color=self.color_cells, alpha=1, linewidth=0.5, method='2d',
                                   view=('x', "z"),
                                   group_neurons=True, rasterize=True)

        #define keyword used in filename if brainmeshes are plotted


        # ax.set_ylim(-700, -200)
        os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("pdf"),exist_ok=True)
        os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("png"),exist_ok=True)
        os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("svg"),exist_ok=True)
        plt.savefig(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("pdf").joinpath(rf"neurotransmitter{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=1200)
        print("PDF saved!")
        plt.savefig(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("png").joinpath(rf"neurotransmitter{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"), dpi=1200)
        print("PNG saved!")
        plt.savefig(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("svg").joinpath(rf"neurotransmitter{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=1200)
        print("SVG saved!")

    def make_interactive(self,show_brs=True):
        if show_brs:
            brkw = "_with_brs_"
        else:
            brkw = "_without_brs_"
        if show_brs:
            brain_meshes = load_brs(self.path_to_data, load_FK_regions=True)
            selected_meshes = ["Retina", 'Midbrain', "Forebrain", "Habenula", "Hindbrain", "Spinal Cord"]
            brain_meshes = [mesh for mesh in brain_meshes if mesh.name in selected_meshes]
            color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)

            fig = navis.plot3d(self.visualized_cells + brain_meshes, backend='plotly',
                               color=self.color_cells + color_meshes, width=1920, height=1080)
            fig.update_layout(
                scene={
                    'xaxis': {'autorange': 'reversed', 'range': (0, 621 * 0.798)},  # reverse !!!
                    'yaxis': {'range': (0, 1406 * 0.798)},

                    'zaxis': {'range': (0, 138 * 2)},
                }
            )

            os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("html"), exist_ok=True)

            plotly.offline.plot(fig, filename=str(
                Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("html").joinpath(
                    f"interactive{brkw[-1]}{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.html")),
                                auto_open=False, auto_play=False)
        else:
            fig = navis.plot3d(self.visualized_cells, backend='plotly',
                               color=self.color_cells, width=1920, height=1080)
            fig.update_layout(
                scene={
                    'xaxis': {'autorange': 'reversed', 'range': (0, 621 * 0.798)},  # reverse !!!
                    'yaxis': {'range': (0, 1406 * 0.798)},

                    'zaxis': {'range': (0, 138 * 2)},
                }
            )

            os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("html"), exist_ok=True)

            plotly.offline.plot(fig, filename=str(
                Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("html").joinpath(
                    f"interactive{brkw[-1]}{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.html")),
                                auto_open=False, auto_play=False)

    def plot_neurotransmitter(self, show_na = True):

        inhibitory_count = 0
        excitatory_count =0
        na_count = 0

        for i,cell in self.all_cells.iterrows():
            if "inhibitory" in cell.cell_type_labels:
                inhibitory_count += 1
            elif "excitatory" in cell.cell_type_labels:
                excitatory_count += 1
            else:
                na_count += 1
        fig = plt.figure()
        plt.bar([1], [inhibitory_count], color="orange")
        plt.bar([2], [excitatory_count], color="blue")
        plt.bar([3], [na_count], color="gray")
        plt.xticks([1, 2, 3], ["I", "E", "NA"])
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


        os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("pdf"),
                    exist_ok=True)
        os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("png"),
                    exist_ok=True)
        os.makedirs(Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("svg"),
                    exist_ok=True)
        fig.savefig(
            Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("pdf").joinpath(
                rf"neurotransmitter{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=1200)
        print("PDF saved!")
        fig.savefig(
            Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("png").joinpath(
                rf"neurotransmitter{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"), dpi=1200)
        print("PNG saved!")
        fig.savefig(
            Path(os.getcwd()).joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("svg").joinpath(
                rf"neurotransmitter{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=1200)
        print("SVG saved!")
        
        

if __name__ == "__main__":
    test_figure = make_figures_FK()

    test_figure.plot_neurotransmitter()

    #test_figure.plot_z_projection()
    #test_figure.plot_z_projection(show_brs=True)
    #test_figure.plot_y_projection()
    #test_figure.make_interactive()
