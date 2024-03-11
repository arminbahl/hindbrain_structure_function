import trimesh as tm
print(tm.__version__)

axon = tm.load("/Users/arminbahl/Desktop/test/clem_zfish1_576460752707815861_axon_mapped.obj")
dendrite = tm.load("/Users/arminbahl/Desktop/test/clem_zfish1_576460752707815861_dendrite_mapped.obj")

#combined = axon.union(dendrite, engine='blender')  ## error in new trimesh
combined = tm.util.concatenate([axon, dendrite])
combined.export("/Users/arminbahl/Desktop/test/combined.obj")
