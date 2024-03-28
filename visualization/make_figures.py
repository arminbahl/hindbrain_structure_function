
# Creates figures for mapped neurons 
# Install environnement using conda env create --file make_videos.yaml
# Version: 0.2 27/03/2024 jbw

import os
import navis
import matplotlib.pyplot as plt

root_cells='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/all_cells/'
root_meshes='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/meshes_zbrain/'

#Find the list of cells to plot
# Function to check if all specified strings are contained in the file
def contains_all_strings(file_path, strings):
    with open(file_path, 'r') as file:
        content = file.read()
        for string in strings:
            if string not in content:
                return False
    return True

# Function to loop over folders and search for text files containing all specified strings
def search_for_all_strings(root_folder, strings):
    matching_folders = []
    for root, dirs, files in os.walk(root_folder):
        for file_name in files:
            if file_name.endswith('.txt'):
                file_path = os.path.join(root, file_name)
                if contains_all_strings(file_path, strings):
                    containing_folder = os.path.dirname(file_path)
                    matching_folders.append(containing_folder)
    return matching_folders
strings_to_search = ['integrator', 'contralateral']  # List of strings to search for
matching_folders = search_for_all_strings(root_cells, strings_to_search)

# Function to load .obj files containing specified keywords from a list of folders
def load_meshes_from_folders(folder_list, keywords):
    loaded_meshes = {keyword: [] for keyword in keywords}  # Initialize dictionary to store meshes for each keyword
    for folder in folder_list:
        mapped_folder = os.path.join(folder, 'mapped')
        if os.path.exists(mapped_folder):
            for file_name in os.listdir(mapped_folder):
                if file_name.endswith('.obj'):
                    for keyword in keywords:
                        if keyword in file_name:
                            file_path = os.path.join(mapped_folder, file_name)
                            mesh = navis.read_mesh(file_path)
                            loaded_meshes[keyword].append(mesh)
                            print(f"Loaded mesh from '{file_path}' containing '{keyword}'.")
    return loaded_meshes
keywords_to_search = ['axon', 'soma', 'dendrite']  # Keywords to search for in file names
loaded_meshes = load_meshes_from_folders(matching_folders, keywords_to_search)

# Now you have a dictionary of loaded meshes separated by keywords
axon_meshes = loaded_meshes['axon']
soma_meshes = loaded_meshes['soma']
dendrite_meshes = loaded_meshes['dendrite']
all_meshes= soma_meshes[1] + dendrite_meshes[1] + axon_meshes[1]

#Make a color with dictionnary
colors_soma = [(0.941, 0.122, 0.122, 1)] * len(matching_folders)
colors_dendrites = [(0.941, 0.122, 0.122, 1)] * len(matching_folders)
colors_axons = [(0, 0, 0, 1)] * len(matching_folders)
all_colors= colors_soma + colors_dendrites + colors_axons

## Make 3 subplots, coronal, dorsal views and dorsal+brain outlines, and E/I ratio 
# Coronal view
fig,ax=navis.plot2d(all_meshes, method='2d', color=all_colors, view=('x', 'z'))
fig.savefig('IC_coronal.png', dpi=1200) 

# Dorsal view
fig,ax=navis.plot2d(all_meshes, method='2d', color=all_colors, view=('x', '-y'))
fig.savefig('IC_dorsal.png', dpi=1200) 

# Dorsal view + brain outlines
meshes_regions=[navis.read_mesh((root_meshes + 'Retina.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Midbrain.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Forebrain.obj'), units='microns'), 
        navis.read_mesh((root_meshes + 'Hindbrain.obj'), units='microns'),
]
color_regions = [(0.4, 0.4, 0.4, 0.1)] * len(meshes_regions)

meshes_and_cells=[meshes_regions, all_meshes]
color_meshes_and_cells=color_regions + all_colors

fig,ax=navis.plot2d(meshes_and_cells, method='2d', color=color_meshes_and_cells, view=('x', '-y'))
ax.set_ylim(-700, -200); 
fig.savefig('IC_dorsal+outlines.png', dpi=1200) 
fig.show()
