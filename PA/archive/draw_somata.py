import pandas as pd
import trimesh as tm
import trimesh as tm
import navis
import skeletor as sk
import tomllib
import tempfile
import os

# Read it as a dataframe for further processing
df_swc = pd.read_csv("/Volumes/ag-bahl_imaging_data3/M11 2P mircroscopes/Florian/function-neurotransmitter-morphology/all_swc/2023-03-27_17-46-50-000_registered_metadata.swc", sep=" ", names=["node_id", 'label', 'x', 'y', 'z', 'radius', 'parent_id'], comment='#', header=None)
soma_x = df_swc.loc[0, "x"]
soma_y = df_swc.loc[0, "y"]
soma_z = df_swc.loc[0, "z"]

sphere = tm.creation.icosphere(radius=2, subdivisions=2)
sphere.apply_translation((soma_x, soma_y, soma_z))
sphere.export("/Volumes/ag-bahl_imaging_data3/M11 2P mircroscopes/Florian/function-neurotransmitter-morphology/all_obj/2023-03-27_17-46-50-000_soma_mapped.obj")


mesh = tm.load_mesh("/Volumes/ag-bahl_imaging_data3/M11 2P mircroscopes/Florian/function-neurotransmitter-morphology/all_obj/2023-03-27_17-46-50-000_mapped.obj")
mesh = sk.pre.fix_mesh(mesh, fix_normals=True, inplace=False)

# Skeletonize the axons and dendrites at 1.5 um precision
skel = sk.skeletonize.by_teasar(mesh, inv_dist=1.5)

# Remove some potential perpedicular branches that should not be there
skel = sk.post.clean_up(skel)
skel = skel.reindex()

# Continue with the swc as a pandas dataframe
df_swc = skel.swc

# Make a standard axon/dendrite radius of 0.5 um everywhere
df_swc["radius"] = 0.5
df_swc["label"] = 0

# Repair gaps in the skeleton using navis
x = navis.read_swc(df_swc)
x = navis.heal_skeleton(x, method='ALL', max_dist=None, min_size=None, drop_disc=False, mask=None, inplace=False)

# Save as a temporary swc
f_swc_temp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.swc')
f_swc_temp.close()

x.to_swc(f_swc_temp.name)

# Read it as a dataframe for further processing
df_swc = pd.read_csv(f_swc_temp.name, sep=" ", names=["node_id", 'label', 'x', 'y', 'z', 'radius', 'parent_id'], comment='#', header=None)

# Delete temporary file
os.remove(f_swc_temp.name)

# Reorder columns for proper storage
df_swc = df_swc.reindex(columns=['node_id', 'label', "x", "y", "z", 'radius', 'parent_id'])

metadata = dict({"test": 4})
metadata["presynapses"] = []
metadata["postsynapses"] = []

# Save the swc
header = (f"# SWC format file based on specifications at http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html\n"
          f"# Generated by 'map_and_skeletonize_cell' of the ANTs registration helper library developed by the Bahl lab Konstanz.\n"
          f"# Metadata: {str(metadata)}\n"
          f"# Labels: 0 = undefined; 1 = soma; 2 = axon; 3 = dendrite; 4 = Presynapse; 5 = Postsynapse\n")

with open("/Volumes/ag-bahl_imaging_data3/M11 2P mircroscopes/Florian/function-neurotransmitter-morphology/all_obj/2023-03-27_17-46-50-000_mapped___test.swc", 'w') as fp:
    fp.write(header)
    df_swc.to_csv(fp, index=False, sep=' ', header=None)