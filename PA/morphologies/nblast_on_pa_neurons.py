import navis
import numpy
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os
import plotly
from analysis_helpers.analysis.personal_dirs.Florian.clem_paper.create_metadata_swc import fix_duplicates
import h5py

def correct_swc(file_name):
    # Repair some strange swc coming from SNT
    f_swc = open(file_name, "r")
    temp = f_swc.read()
    f_swc.close()

    temp = temp.replace("\t", " ")
    if "# Exported from SNT" in temp:
        f_swc = open(file_name, "w")
        f_swc.write(temp)
        f_swc.close()

        cell_data = np.loadtxt(file_name, skiprows=5, delimiter=' ', ndmin=2)

        transformed_cell_data = np.c_[cell_data[:, 0],
        cell_data[:, 1],
        cell_data[:, 0],
        cell_data[:, 1],
        cell_data[:, 2],
        cell_data[:, 5],
        cell_data[:, 6]]

        df = pd.DataFrame(cell_data, columns=['index', 'neuronname', 'x', 'y', 'z', 'size', 'connects'])

        df["index"] = df["index"].astype(np.int64)
        df["neuronname"] = df["neuronname"].astype(np.int64)
        df["connects"] = df["connects"].astype(np.int64)
        os.remove(file_name)
        df.to_csv(file_name, index=False, sep=' ', header=False)

def get_resolution(path_hdf5):
    path = path_hdf5.joinpath(path_hdf5.name+".hdf5")
    print(2)
    with h5py.File(path, 'r') as f:
        dx = f['repeat00_tile000_z000_950nm/raw_data/fish00/imaging_data_channel0'].attrs['dx']
        dy = f['repeat00_tile000_z000_950nm/raw_data/fish00/imaging_data_channel0'].attrs['dy']
        dz = f.attrs['z_scan_abs_dz']
    return dx,dy,dz

def shift_cell(cell, target_cell):
    shift_x = target_cell.nodes.loc[0,'x'] - cell.nodes.loc[0,'x']
    shift_y = target_cell.nodes.loc[0, 'y'] - cell.nodes.loc[0, 'y']
    shift_z = target_cell.nodes.loc[0, 'z'] - cell.nodes.loc[0, 'z']

    cell.nodes.loc[:,['x','y','z']] = cell.nodes.loc[:,['x','y','z']] + np.array([shift_x, shift_y, shift_z])
    return cell

def scale_cell(cell,cell_dx_dy, target_cell_dx_dy):
    scale_factor = target_cell_dx_dy/cell_dx_dy
    cell.nodes.loc[:,['x','y']] = cell.nodes.loc[:,['x','y']] * np.array([scale_factor, scale_factor])
    return cell

def to_micron(cell,dx_dy_dz):
    cell.nodes.loc[:, ['x', 'y', 'z']] = cell.nodes.loc[:, ['x', 'y', 'z']] * np.array([dx_dy_dz[0], dx_dy_dz[1], dx_dy_dz[2]])
    return cell
if __name__ == '__main__':
    neurons = []

    neuron_mesh = []
    neuron_names = []
    one_letter_neuron_type = []
    cell_register = pd.read_csv(r"C:\Users\ag-bahl\Downloads\Table1.csv")
    cell_register = cell_register.iloc[5:,:]
    cell_register = cell_register.loc[~cell_register['ROI'].isna(),:]

    for file in os.listdir(r'W:\Florian\function-neurotransmitter-morphology') :

        if Path(rf'W:\Florian\function-neurotransmitter-morphology\{file}\{file}-000_registered.swc').exists() and file in list(cell_register['Volume']):

            neuron = navis.read_swc(rf'W:\Florian\function-neurotransmitter-morphology\{file}\{file}-000_registered.swc')
            neuron.soma = 1
            neuron.nodes.iloc[0, 5] = 2
            neuron = to_micron(neuron, [0.798, 0.798, 2])
            neuron = navis.smooth_skeleton(neuron)
            neuron = navis.resample_skeleton(neuron, 0.1)
            neuron.units = 'microns'
            neuron.nodes = fix_duplicates(neuron.nodes)

            neurons.append(neuron)
            one_letter_neuron_type.append(cell_register.loc[cell_register['Volume']==file,"Manually evaluated cell type"].values[0][0])
            neuron_names.append(file)
            # mesh = navis.conversion.tree2meshneuron(neuron, use_normals=True, tube_points=20)
            # neuron_mesh.append(mesh)



    #plot skeleton

    viewer = navis.plot3d(neurons,backend='plotly', inline=True,width=1920,height=1080)
    viewer.update_layout(
        scene={
            'xaxis': {'autorange':True},  # reverse !!!
            'yaxis': {'autorange':True},

            'zaxis': {'autorange':True},
            'aspectmode':"data",
            'aspectratio':{"x": 1, "y":1, "z":1}
        }
    )
    plotly.offline.plot(viewer, filename=rf"C:\Users\ag-bahl\PycharmProjects\analysis_helpers\analysis\personal_dirs\Florian\interindividual_tracing_comparison\plots\nblast\skeleton.html",auto_open=True,auto_play=False)


    #plot mesh
    viewer = navis.plot3d(neuron_mesh, backend='plotly', inline=True, width=1920, height=1080)
    viewer.update_layout(
        scene={
            'xaxis': {'autorange': True},  # reverse !!!
            'yaxis': {'autorange': True},

            'zaxis': {'autorange': True},
            'aspectmode': "data",
            'aspectratio': {"x": 1, "y": 1, "z": 1}
        }
    )
    plotly.offline.plot(viewer, filename=rf"C:\Users\ag-bahl\PycharmProjects\analysis_helpers\analysis\personal_dirs\Florian\interindividual_tracing_comparison\plots\nblast\mesh.html", auto_open=False,
                        auto_play=False)
    import trimesh as tm
    scene = tm.Scene(neuron_mesh)
    scene.export(fr"C:\Users\ag-bahl\PycharmProjects\analysis_helpers\analysis\personal_dirs\Florian\interindividual_tracing_comparison\plots\nblast\mesh.obj")

    #NBLAST

    my_neuron_list = navis.NeuronList(neurons)
    dps = navis.make_dotprops(my_neuron_list, k=5, resample=False)

    nbl = navis.nblast(dps,dps, progress=False)
    nbl.index = neuron_names
    nbl.columns = neuron_names

    nbl_array = np.array(nbl)


    #hirarchical clustering
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, dendrogram, set_link_color_palette

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcl
    import seaborn as sns

    plt.figure(figsize=(20,20))
    nbl_mean = (nbl + nbl.T) / 2
    nbl_dist = 1 - nbl_mean

    set_link_color_palette([mcl.to_hex(c) for c in sns.color_palette('muted', 10)])

    # To generate a linkage, we have to bring the matrix from square-form to vector-form
    nbl_vec = squareform(nbl_dist, checks=False)

    # Generate linkage
    Z = linkage(nbl_vec, method='ward')

    # Plot a dendrogram
    dn = dendrogram(Z, labels=one_letter_neuron_type,color_threshold=1)

    ax = plt.gca()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right',fontdict={'fontsize':20})

    sns.despine(trim=True, bottom=True)
    plt.show()



    #all

    my_neuron_list = navis.NeuronList(neurons)
    dps = navis.make_dotprops(my_neuron_list, k=5, resample=False)

    # Run an all-by-all NBLAST and synNBLAST
    pns_nbl = navis.nblast_allbyall(dps, progress=False)


    # Generate the linear vectors
    nbl_vec = squareform(((pns_nbl + pns_nbl.T) / 2 - 1) * -1, checks=False)


    # Generate linkages
    Z_nbl = linkage(nbl_vec, method='ward', optimal_ordering=True)


    # Plot dendrograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dn1 = dendrogram(Z_nbl, no_labels=True, color_threshold=1, ax=axes[0])


    axes[0].set_title('NBLAST')


    sns.despine(trim=True, bottom=True)

    # Generate clusters
    from scipy.cluster.hierarchy import fcluster

    cl = fcluster(Z_nbl, t=1, criterion='distance')

    # Now plot each cluster
    # For simplicity we are plotting in 2D here
    import math

    n_clusters = max(cl)
    rows = 4
    cols = math.ceil(n_clusters / 4)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(20, 5 * cols))
    # Flatten axes
    axes = [ax for l in axes for ax in l]

    # Generate colors
    pal = sns.color_palette('muted', n_clusters)

    for i in range(n_clusters):
        ax = axes[i]
        ax.set_title(f'cluster {i + 1}')
        # Get the neurons in this cluster
        this = my_neuron_list[cl == (i + 1)]

        navis.plot2d(this, method='2d', ax=ax, color=pal[i], lw=1.5, view=('x', '-z'), alpha=.5)

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_axis_off()

        # Set all axes to the same limits
        bbox = my_neuron_list.bbox
        ax.set_xlim(bbox[0][0], bbox[0][1])
        ax.set_ylim(-bbox[2][1], -bbox[2][0])

    cluster_df = pd.DataFrame({'type':one_letter_neuron_type,'cluster':cl})
