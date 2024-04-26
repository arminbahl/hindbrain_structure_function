import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
import numpy as np
import navis
from pathlib import Path
import pandas as pd
import matplotlib as mpl
from hindbrain_structure_function.visualization.FK_tools.load_pa_table import *
from hindbrain_structure_function.visualization.FK_tools.load_clem_table import *
from hindbrain_structure_function.visualization.FK_tools.load_mesh import *
from hindbrain_structure_function.visualization.FK_tools.load_brs import *
from hindbrain_structure_function.visualization.FK_tools.get_base_path import *
from datetime import datetime
import plotly
import matplotlib
import warnings
warnings.filterwarnings("ignore")

matplotlib.use('TkAgg')
class make_figures_FK:
    """
    A class for generating and saving various visualizations of brain cells based on neurotransmitter types, spatial projections, and interactive 3D models.

    This class provides functionalities to create visual representations of brain cells, categorizing them based on their neurotransmitter types (inhibitory, excitatory, or unspecified). It also allows for the generation of 2D spatial projections (Y and Z axes) and interactive 3D plots of the cells, with options to include selected brain regions for enhanced context.

    Upon initialization, the class loads and processes cell data from specified modalities (e.g., 'pa' for photoactivation, 'clem' for correlative light and electron microscopy) and keywords (e.g., 'integrator', 'contralateral') to filter the cells of interest. It then loads the mesh data for each cell that fits the queries in the selected modalities.

    The visualizations generated by this class can include selected brain regions, differentiated cell types by color, and can be saved in various formats including HTML (for interactive 3D plots), PDF, PNG, and SVG.

    Parameters:
    - modalities (list of str): The imaging modalities to load cell data from. Currently supports 'pa' and 'clem'. Default is ['pa'].
    - keywords (list of str): Keywords to filter the cells by their type labels. Default is ['integrator', 'contralateral'].

    Attributes:
    - name_time (datetime): Timestamp used for naming saved files to ensure uniqueness.
    - path_to_data (Path): The path to the directory containing cell and mesh data.
    - all_cells (DataFrame): The loaded and processed DataFrame containing cell data after filtering by modalities and keywords.

    Methods:
    - plot_z_projection(show_brs=False, force_new_cell_list=False, ylim=[-700, -200]): Generates and saves a 2D Z-axis projection plot.
    - plot_y_projection(show_brs=False, force_new_cell_list=False): Generates and saves a 2D Y-axis projection plot.
    - make_interactive(show_brs=True): Generates and saves an interactive 3D plot of visualized brain cells.
    - plot_neurotransmitter(show_na=True): Generates and saves a bar chart showing the count of cells by neurotransmitter type.

    Usage:
    To use this class, initialize an instance with the desired modalities and keywords, and then call its methods to generate and save the visualizations:

    ```python
    figure_maker = make_figures_FK(modalities=['pa', 'clem'], keywords=['integrator'])
    figure_maker.plot_neurotransmitter()
    figure_maker.plot_z_projection(show_brs=True)
    ```

    Note:
    - This class relies on external libraries such as `navis`, `matplotlib`, `plotly`, and `pandas` for data processing and visualization.
    - Ensure that the path to the data directory is correctly set up in the `path_configuration.txt` file to successfully load the cell and mesh data.
    """
    def __init__(self,
                 modalities = ['pa'],
                 keywords = ['integrator','ipsilateral'],
                 use_smooth_pa=True,
                 mirror=True):
        
        #set_name_time
        self.name_time = datetime.now()
        self.use_smooth_pa = use_smooth_pa
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
        self.keywords = keywords
        #subset dataset for keywords
        if keywords!='all':
            for keyword in keywords:
                subset_for_keyword = all_cells['cell_type_labels'].apply(lambda current_label: True if keyword.replace("_"," ") in current_label or keyword in current_label  else False)
                all_cells = all_cells[subset_for_keyword]


        self.selected_meshes = ["Retina", 'Midbrain',"Olfactory Bulb", "Forebrain", "Habenula", "Hindbrain", "Spinal Cord","raphe",'eye1','eye2','cerebellar_neuropil','Rhombencephalon - Rhombomere 1','Rhombencephalon - Rhombomere 2',"cn1","cn2",'Mesencephalon - Tectum Stratum Periventriculare']
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
            all_cells.loc[i,:] = load_mesh(cell,self.path_to_data,use_smooth_pa=self.use_smooth_pa)

        width_brain = 495.56
        if mirror:
            for i,cell in all_cells.iterrows():
                if np.mean(cell['soma_mesh']._vertices[:,0]) > (width_brain/2): #check which hemisphere
                    all_cells.loc[i, 'soma_mesh']._vertices = navis.transforms.mirror(cell['soma_mesh']._vertices, width_brain, 'x')
                    if cell['imaging_modality'] == 'photoactivation':
                        all_cells.loc[i,'neurites_mesh']._vertices = navis.transforms.mirror(cell['neurites_mesh']._vertices,width_brain,'x')
                    if cell['imaging_modality'] == 'clem':
                        all_cells.loc[i,'axon_mesh']._vertices = navis.transforms.mirror(cell['axon_mesh']._vertices,width_brain,'x')
                        all_cells.loc[i, 'dendrite_mesh']._vertices = navis.transforms.mirror(cell['dendrite_mesh']._vertices, width_brain, 'x')




        self.all_cells = all_cells

        os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("used_cells"), exist_ok=True)
        all_cells['cell_name'].to_csv(self.path_to_data.joinpath("make_figures_FK_output").joinpath("used_cells").joinpath(f'{"_".join(self.keywords)}_{self.name_time.strftime("%Y-%m-%d_%H-%M-%S")}.txt'), index=False, header=None)

    def plot_z_projection(self, show_brs=False, force_new_cell_list=False, ylim=[-700, -200],rasterize=True,black_neuron = True,standard_size=True,volume_outlines=True,background_gray=True):
        """
        Generates and saves a 2D Z-axis projection plot of visualized brain cells, with an option to include selected brain regions.

        This function creates a 2D plot showing the Z-axis projection of brain cells, differentiating cell types by color. It allows for the optional inclusion of specific brain regions in the visualization. If `force_new_cell_list` is True, or no list of visualized cells exists, it compiles a new list based on the cell types present in the dataset and their corresponding meshes.

        Parameters:
        - show_brs (bool): If True, selected brain regions will be included in the plot alongside the brain cells. Defaults to False.
        - force_new_cell_list (bool): If True, forces the creation of a new list of visualized cells and their colors, ignoring any pre-existing list. Defaults to False.
        - ylim (list of int): A two-element list specifying the y-axis limits for the plot. This controls the vertical extent of the visualization. Defaults to [-700, -200].

        The function automatically creates the necessary directories for saving the plots and saves the plot in PNG, PDF, and SVG formats. The saved files include a timestamp and an indicator of whether brain regions were included, ensuring unique filenames for each execution.

        Notes:
        - The cell type and their corresponding colors are defined within the function in a dictionary.
        - This function relies on the 'navis' library for generating the plot.
        - The visualization's focus is customizable through the `ylim` parameter, allowing users to adjust the view to specific areas of interest.

        Example usage:
            instance.plot_z_projection(show_brs=True, force_new_cell_list=False, ylim=[-700, -200])
        This will generate and save a Z-axis projection plot including brain regions, with the specified y-axis limits.
        """
        #define colors for cells
        color_cell_type_dict = {"integrator": (255, 0, 0.7),
                                "dynamic threshold": (0, 255, 255, 0.7),
                                "motor command": (128, 0, 128, 0.7), }

        #load needed meshes into a list and assign colors

        if not "visualized_cells" in self.__dir__() or force_new_cell_list:
            self.visualized_cells = []
            self.color_cells = []

            for i,cell in self.all_cells.iterrows():
                       if black_neuron==True and cell["imaging_modality"] == "photoactivation":
                           self.color_cells.append("black")
                           self.color_cells.append("black")
                           black_neuron=False
                       elif type(black_neuron) == str:
                           if cell['cell_name'] == black_neuron:
                               self.color_cells.append("black")
                               self.color_cells.append("black")
                       else:
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



        #here we start the plotting
        if show_brs:
            brain_meshes = load_brs(self.path_to_data, load_FK_regions=True)
            brain_meshes_with_vertices = load_brs(self.path_to_data, load_FK_regions=True, as_volume=False)
            selected_meshes = self.selected_meshes
            brain_meshes = [mesh for mesh in brain_meshes if mesh.name in selected_meshes]
            brain_meshes_with_vertices = [mesh for mesh in brain_meshes_with_vertices if mesh.name in selected_meshes]
            color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)

            for i, mesh in enumerate(brain_meshes):
                brain_meshes[i]._vertices = brain_meshes_with_vertices[i]._vertices
                brain_meshes[i]._faces = brain_meshes_with_vertices[i]._faces

        #zprojection
        if show_brs:

            plt.ylim(ylim)



            fig, ax = navis.plot2d(brain_meshes, color=color_meshes, volume_outlines=volume_outlines, alpha=0.2, linewidth=0.5, method='2d', view=('x', "-y"), group_neurons=True, rasterize=rasterize)
            if background_gray:


                for mesh in brain_meshes:
                     temp_convex_hull = np.array(mesh.to_2d(view=('x', "-y")))

                     ax.fill(temp_convex_hull[:,0],temp_convex_hull[:,1],c='#F7F7F7',zorder=-1,alpha=1,ec=None)

            ax.axvline(250, color=(0.85, 0.85, 0.85, 0.2), linestyle='--', alpha=0.5,lw=1)
            fig, ax = navis.plot2d(self.visualized_cells,color=self.color_cells,alpha=1,linewidth=0.5,method='2d',view=('x', "-y"),group_neurons=True,rasterize=rasterize,ax=ax, scalebar="20 um")


            if standard_size:
                width_brain = 495.56
                plt.xlim(0,width_brain)
                plt.ylim( -850,-50) #minimum of forebrain and maximum of hindbrain


        else:

            fig, ax = navis.plot2d(self.visualized_cells, color=self.color_cells, alpha=1, linewidth=0.5, method='2d', view=('x', "-y"),
                                   group_neurons=True, rasterize=rasterize, scalebar="10 um")
            ax.axvline(250, color=(0.85, 0.85, 0.85, 0.2), linestyle='--', alpha=0.5,zorder=0)
            if standard_size:
                width_brain = 495.56
                plt.xlim(0,width_brain)
                plt.ylim( -850,-50) #minimum of forebrain and maximum of hindbrain



        if show_brs:
            brkw = "_with_brs_"
        else:
            brkw = "_without_brs_"

        #ax.set_ylim(-700, -200)
        os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("pdf"),exist_ok=True)
        os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("png"),exist_ok=True)
        os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("svg"),exist_ok=True)
        fig.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("png").joinpath(rf"z_projection{brkw}{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"), dpi=1200)
        fig.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("pdf").joinpath(rf"z_projection{brkw}{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=1200)
        fig.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("z_projection").joinpath("svg").joinpath(rf"z_projection{brkw}{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.svg"), dpi=1200)
        print("Z projection saved!")

    def plot_y_projection(self, show_brs=False, force_new_cell_list=False,rasterize = True,black_neuron=True,standard_size=True,volume_outlines=True,background_gray=True):
        """
        Generates and saves a 2D Y-axis projection plot of visualized brain cells, optionally including selected brain regions.

        This function creates a 2D plot representing the Y-axis projection of brain cells and, if specified, selected brain regions.
        The plot differentiates cell types based on color coding specified in a dictionary within the function. If `force_new_cell_list`
        is True, or if no cell list is pre-loaded, the function will generate a new list of visualized cells and their associated colors
        based on cell type labels.

        The inclusion of brain regions in the plot is controlled by the `show_brs` flag. If enabled, specific brain regions are loaded
        and visualized alongside the brain cells. The plot is saved in multiple formats (PDF, PNG, SVG) within a structured directory
        hierarchy based on the plot type and timestamp.

        Parameters:
        - show_brs (bool): Controls whether selected brain regions are included in the plot alongside the visualized cells.
                           Defaults to False.
        - force_new_cell_list (bool): Forces the generation of a new list of visualized cells and their colors if True, regardless
                                      of whether a list already exists. Defaults to False.

        The function creates necessary directories if they do not exist and saves the plot in PDF, PNG, and SVG formats, each
        within its respective sub-directory under "make_figures_FK_output/y_projection". The filenames include a timestamp, ensuring
        unique filenames for each execution.

        Note:
        - This function relies on the 'navis' library for plotting and assumes the presence of a pre-defined structure of cell data
          and brain region data within the class instance.

        Example usage:
            instance.plot_y_projection(show_brs=True, force_new_cell_list=False)
        This generates a 2D Y-axis projection plot including brain regions and saves it in specified formats.
        """
        # define colors for cells
        color_cell_type_dict = {"integrator": (255,0,0.7),
                                "dynamic threshold": (0,255,255,0.7),
                                "motor command": (128,0,128,0.7), }

        # load needed meshes into a list and assign colors

        if not "visualized_cells" in self.__dir__() or force_new_cell_list:
            self.visualized_cells = []
            self.color_cells = []
            for i, cell in self.all_cells.iterrows():
                if black_neuron == True and cell["imaging_modality"] == "photoactivation":
                    self.color_cells.append("black")
                    self.color_cells.append("black")
                    black_neuron = False
                elif type(black_neuron) == str:
                    if cell['cell_name'] == black_neuron:
                        self.color_cells.append("black")
                        self.color_cells.append("black")
                else:
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
            brain_meshes_with_vertices = load_brs(self.path_to_data, load_FK_regions=True,as_volume=False)
            selected_meshes = self.selected_meshes
            brain_meshes = [mesh for mesh in brain_meshes if mesh.name in selected_meshes]
            brain_meshes_with_vertices = [mesh for mesh in brain_meshes_with_vertices if mesh.name in selected_meshes]
            color_meshes = [(0.4, 0.4, 0.4, 0.1)] * len(brain_meshes)

            for i,mesh in enumerate(brain_meshes):
                brain_meshes[i]._vertices = brain_meshes_with_vertices[i]._vertices
                brain_meshes[i]._faces = brain_meshes_with_vertices[i]._faces

        # yprojection
        if show_brs:


            fig, ax = navis.plot2d(brain_meshes, color=color_meshes, volume_outlines=volume_outlines, alpha=0.2, linewidth=0.5, method='2d', view=('x', "z"), group_neurons=True, rasterize=rasterize)

            if background_gray:

                for mesh in brain_meshes:
                    temp_convex_hull = np.array(mesh.to_2d(view=('x', "z")))

                    ax.fill(temp_convex_hull[:, 0], temp_convex_hull[:, 1], c='#F7F7F7', zorder=-1, alpha=1,ec=None)
            ax.axvline(250, color='white', linestyle='--', alpha=0.5)
            fig, ax = navis.plot2d(self.visualized_cells, color=self.color_cells, alpha=1, linewidth=0.5, method='2d',
                                   view=('x', "z"), group_neurons=True, rasterize=rasterize, ax=ax, scalebar="20 um")

            if standard_size:
                width_brain = 495.56
                plt.xlim(0,width_brain)
                plt.ylim(0, 270)
            ax.axvline(250, color=(0.85, 0.85, 0.85, 0.2), linestyle='--', alpha=0.5)

        else:
            fig, ax = navis.plot2d(self.visualized_cells, color=self.color_cells, alpha=1, linewidth=0.5, method='2d',
                                   view=('x', "z"),
                                   group_neurons=True, rasterize=rasterize, scalebar="10 um")
            ax.axvline(248.379, color=(0.85, 0.85, 0.85, 0.2), linestyle='--', alpha=0.5, zorder=0)
            if standard_size:
                width_brain = 495.56
                plt.xlim(0,width_brain)
                plt.ylim(0, 270)

        # ax.set_ylim(-700, -200)
        os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("pdf"),exist_ok=True)
        os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("png"),exist_ok=True)
        os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("svg"),exist_ok=True)
        plt.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("pdf").joinpath(rf"{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=1200)
        plt.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("png").joinpath(rf"{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"), dpi=1200)
        plt.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("y_projection").joinpath("svg").joinpath(rf"{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.svg"), dpi=1200)
        print("Y projection saved!")

    def make_interactive(self, show_brs=True):
        """
        Generates and saves an interactive 3D plot of visualized brain cells, with an option to include selected brain regions.

        This function uses the Plotly library to create a dynamic, interactive 3D visualization of brain cells. Users can
        opt to include a predefined set of brain regions (e.g., Retina, Midbrain, Forebrain) in the visualization. The
        appearance of the plot, including the orientation and scale of the axes, is configured for optimal viewing. The
        generated plot is saved as an HTML file, facilitating easy sharing and viewing in web browsers.

        Parameters:
        - show_brs (bool): A flag to determine whether to include brain regions in the visualization. If True, the function
                           loads and visualizes selected brain regions alongside the brain cells. If False, only brain cells
                           are visualized. Defaults to True.

        The output file is saved in a directory named "make_figures_FK_output/html" within the current working directory.
        The filename includes a timestamp and indicates whether brain regions were included in the plot.

        Note:
        - The function creates the necessary directories if they do not exist.
        - The 3D plot is saved but not automatically opened, allowing users to view it at their convenience.

        Example usage:
            instance.make_interactive(show_brs=True)
        This will generate a 3D plot including brain regions and save it as an HTML file.
        """
        black_neuron = False
        color_cell_type_dict = {"integrator": (255, 0, 0.7),
                                "dynamic threshold": (0, 255, 255, 0.7),
                                "motor command": (128, 0, 128, 0.7), }

        if not "visualized_cells" in self.__dir__():
            self.visualized_cells = []
            self.color_cells = []

            for i,cell in self.all_cells.iterrows():
                       if black_neuron==True and cell["imaging_modality"] == "photoactivation":
                           self.color_cells.append("black")
                           self.color_cells.append("black")
                           black_neuron=False
                       elif type(black_neuron) == str:
                           if cell['cell_name'] == black_neuron:
                               self.color_cells.append("black")
                               self.color_cells.append("black")
                       else:
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


        if show_brs:
            brkw = "_with_brs_"
        else:
            brkw = "_without_brs_"
        if show_brs:
            brain_meshes = load_brs(self.path_to_data, load_FK_regions=True)
            selected_meshes = self.selected_meshes
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

            os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("html"), exist_ok=True)

            plotly.offline.plot(fig, filename=str(Path(self.path_to_data.joinpath("make_figures_FK_output").joinpath("html").joinpath(f"interactive{brkw[-1]}{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.html"))),auto_open=False, auto_play=False)
            print("Interactive saved!")

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
            os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("html"), exist_ok=True)


            plotly.offline.plot(fig, filename=str(Path(self.path_to_data.joinpath("make_figures_FK_output").joinpath("html").joinpath(f"interactive{brkw[-1]}{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.html"))),auto_open=False, auto_play=False)
            print("Interactive saved!")

    def plot_neurotransmitter(self, show_na=True):
        """
        Generates and saves a bar chart representing the count of cells classified by neurotransmitter type: inhibitory, excitatory, or unspecified (NA).

        This function counts the number of cells labeled as 'inhibitory', 'excitatory', or not specified (NA) within the dataset. It then generates a bar chart visualizing these counts. The bars are colored distinctly to differentiate between the categories: orange for inhibitory, blue for excitatory, and gray for unspecified. The chart includes an option to include or exclude the NA category based on the `show_na` parameter. After plotting, the figure is saved in PDF, PNG, and SVG formats within designated subdirectories.

        Parameters:
        - show_na (bool): Controls the inclusion of the NA (not specified) category in the plot. If True, the NA category is included; if False, it is excluded from the visualization. Defaults to True.

        The function ensures that the necessary directories for saving the figures are created if they do not already exist. The saved figures include a timestamp in their filenames to avoid overwriting previous files.

        Notes:
        - The Y-axis of the bar chart is set to display integer values only to enhance readability.
        - The function relies on matplotlib for plotting and assumes the presence of a 'all_cells' DataFrame attribute within the class instance, where cell types are categorized.

        Example usage:
          instance.plot_neurotransmitter(show_na=True)
        This will generate and save a bar chart including the unspecified (NA) category, representing the count of cells by neurotransmitter type.
        """

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
        # fig = plt.figure()
        # plt.bar([1], [inhibitory_count], color="orange",width =0.9)
        # plt.bar([2], [excitatory_count], color="blue",width =0.9)
        # plt.bar([3], [na_count], color="gray",width =0.9)
        # plt.xticks([1, 2, 3], ["I", "E", "NA"])
        # ax = plt.gca()
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))


        os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("pdf"),exist_ok=True)
        os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("png"),exist_ok=True)
        os.makedirs(self.path_to_data.joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("svg"),exist_ok=True)

        #stacked bar plot

        cmap = cm.get_cmap('Greys')
        g1,g2,g3 = cmap(0.8),cmap(0.6),cmap(0.4)

        width =1
        colors = [g1,g2,g3]
        all_nt = [inhibitory_count,excitatory_count,na_count]
        texts = ['Gad1b+',"Vglut2a+", "Unclear"]
        fig, ax = plt.subplots()
        current_position = 0
        for nt_count,color,text in zip(all_nt, colors,texts):
            if nt_count!=0:
                ax.bar(0,nt_count,bottom = current_position, width=width,color = color)
                ax.text(1, current_position+(nt_count/2),text,ha='left',va='center',c='k')
                ax.text(0, current_position + (nt_count / 2), nt_count,ha='center',va='center',c='white')
                current_position += nt_count + (np.sum(all_nt)*0.01)

        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)


        fig.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("pdf").joinpath(rf"neurotransmitter_stacked_bar_{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=1200)
        fig.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("png").joinpath(rf"neurotransmitter_stacked_bar_{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"), dpi=1200)
        fig.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("svg").joinpath(rf"neurotransmitter_stacked_bar_{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.svg"), dpi=1200)


        #simple bar
        width = 1
        locs = [1.5 * (x * width) for x in range(3)]
        fig, ax = plt.subplots()
        for nt_count,color,text,loc in zip(all_nt, colors,texts,locs):
            ax.bar(loc,nt_count, width=width,color = g3)

            ax.text(loc, (nt_count / 2), nt_count,ha='center',va='center',c='white')
        ax.set_xticks(locs,texts,rotation=45)
        ax.set_yticks([])


        ax.axis('equal')
        ax.spines[['right', 'top','left','bottom']].set_visible(False)
        fig.subplots_adjust(bottom=0.15)
        fig.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("pdf").joinpath(rf"neurotransmitter_simple_gray_{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"), dpi=1200)
        fig.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("png").joinpath(rf"neurotransmitter_simple_gray_{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.png"), dpi=1200)
        fig.savefig(self.path_to_data.joinpath("make_figures_FK_output").joinpath("neurotransmitter").joinpath("svg").joinpath(rf"neurotransmitter_simple_gray_{'_'.join(self.keywords)}_{self.name_time.strftime('%Y-%m-%d_%H-%M-%S')}.svg"), dpi=1200)






        print("Neurotransmitter saved!")
        
        

if __name__ == "__main__":
    integrator_ipsi_figure = make_figures_FK(modalities=['pa'],keywords=['integrator','ipsilateral',])
    integrator_ipsi_figure.plot_z_projection(rasterize=True, show_brs=True, )
    integrator_ipsi_figure.plot_y_projection(show_brs=True, rasterize=True)
    integrator_ipsi_figure.plot_neurotransmitter()

    integrator_contra_figure = make_figures_FK(modalities=['pa'],keywords=['integrator','contralateral',])
    integrator_contra_figure.plot_z_projection(rasterize=True, show_brs=True, )
    integrator_contra_figure.plot_y_projection(show_brs=True, rasterize=True)
    integrator_contra_figure.plot_neurotransmitter()
    
    dt_figure = make_figures_FK(modalities=['pa'],keywords=['dynamic_threshold'],mirror=True)
    dt_figure.plot_z_projection(rasterize=True,show_brs=True,)
    dt_figure.plot_y_projection(show_brs=True, rasterize=True)
    dt_figure.plot_neurotransmitter()

    
    mc_figure = make_figures_FK(modalities=['pa'],keywords=['motor_command'])
    mc_figure.plot_z_projection(rasterize=True, show_brs=True, )
    mc_figure.plot_y_projection(show_brs=True, rasterize=True)
    mc_figure.plot_neurotransmitter()