import navis
import trimesh as tm
from pathlib import Path
path = Path('/Users/fkampf/Downloads/2024-01-22_15-05-28_fish001_KS-000_to_volume_to_ZBRAIN.swc')

neuron = navis.read_swc(path)
neuron.soma = 1          
neuron.nodes.loc[:, 'radius'] = 0.3
neuron_mesh = navis.conversion.tree2meshneuron(neuron, use_normals=True, tube_points=20)
neuron_mesh = tm.Trimesh(
    vertices=neuron_mesh.vertices,
    faces=neuron_mesh.faces,
    process=False   # turn off trimesh’s cleanup if you want the raw mesh
)

smoothed_neuron = navis.smooth_skeleton(neuron,window=20) #previously 7
smoothed_neuron.nodes.iloc[0, :] = neuron.nodes.iloc[0,:]
smoothed_neuron_mesh = navis.conversion.tree2meshneuron(smoothed_neuron, use_normals=True, tube_points=20)
smoothed_neuron_mesh = tm.Trimesh(
    vertices=smoothed_neuron_mesh.vertices,
    faces=smoothed_neuron_mesh.faces,
    process=False   # turn off trimesh’s cleanup if you want the raw mesh
)

x = neuron.nodes.loc[0, "x"]
y = neuron.nodes.loc[0, "y"]
z = neuron.nodes.loc[0, "z"]

sphere = tm.creation.icosphere(radius=2, subdivisions=2)
sphere.apply_translation((x, y, z))
combined_list = [neuron_mesh, sphere]
combined_list_smooth = [smoothed_neuron_mesh, sphere]
scene = tm.Scene(combined_list)
scene_smoothed = tm.Scene(combined_list_smooth)
scene.export(str(path)[:-4]+'.obj')
scene_smoothed.export(str(path)[:-4]+'_smoothed.obj')