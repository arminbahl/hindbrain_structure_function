import numpy as np
import pandas as pd
import navis
from pathlib import Path
import os
import datetime
import copy







#drop rows where swc file is not registered
def check_registered_swc(file_name):
    if Path(rf"W:\Florian\function-neurotransmitter-morphology\{file_name}\{file_name}-000_registered.swc").exists():
        return file_name
    else:
        return np.nan


def fix_duplicates(df):
    import copy

    work_df = copy.deepcopy(df)

    adoptive_parent = {}
    drop_set = set()
    drop_not_set = set()

    for i_outer, item_outer in work_df.iterrows():

            for i,item in work_df.iterrows():
                if item['node_id'] - 1 not in drop_not_set:
                    if (item['x'] == item_outer['x']) and  (item['y'] == item_outer['y']) and  (item['z'] == item_outer['z']) and (item['node_id'] != item_outer['node_id']):
                            if item_outer['node_id'] in adoptive_parent.keys():
                                while item_outer['node_id'] in adoptive_parent.keys():
                                    item_outer['node_id'] = adoptive_parent[item_outer['node_id']]
                            else:
                                adoptive_parent[int(item["node_id"])] = int(item_outer['node_id'])
                            drop_set.add(int(item["node_id"]-1))
                            drop_not_set.add(int(item_outer['node_id']-1))

    print("DROP SET", list(drop_set))

    work_df.drop(list(drop_set), axis=0, inplace=True)
    work_df.loc[:, 'parent_id'].replace(to_replace=adoptive_parent, inplace=True)

    for i,item in work_df.iterrows():
        if item['parent_id'] in adoptive_parent.keys():
            work_df.loc[i, 'parent_id'] = adoptive_parent[item['parent_id']]


    return work_df
def nid(input):
    if input == '-':
        return 'gad1b'
    if input == '+':
        return 'vglut2a'
    if input == "":
        return 'to be identified '

    else:
        return 'na'

if __name__ == '__main__':
    cell_register = pd.read_csv(r"C:\Users\ag-bahl\Downloads\Table1 (1).csv")   #load data
    cell_register = cell_register[~cell_register['Function'].isna()]    #drop empty rows



    for i,item in cell_register.iterrows():
        neuron = navis.read_swc(Path(rf"W:\Florian\function-neurotransmitter-morphology\{item['Volume']}\{item['Volume']}-000_registered.swc"))
        neuron.soma = 1
        neuron.nodes.iloc[0, 5] = 2

        id = item['Internal name']
        name = f"{item['Volume']}-000_registered.swc"
        units = 'microns'
        tracer_name = 'Florian Kaempf'
        classifier = item['Manually evaluated cell type'] + ", " + nid(item["NID"])
        certainty_NID = str(item['certainty of NID assessment'])
        imaging_modality = 'LM - photoactivation'
        date_of_the_tracing = datetime.datetime.fromtimestamp(os.path.getmtime(Path(rf"W:\Florian\function-neurotransmitter-morphology\{item['Volume']}\{item['Volume']}-000_registered.swc"))).strftime("%Y-%m-%d_%H-%M-%S")
        soma_position = f'{neuron.nodes.loc[0,"x"]},{neuron.nodes.loc[0,"y"]},{neuron.nodes.loc[0,"z"]}'

        my_meta = {'id':str(id),
                'name':name,
                'units':units,
                'tracer_name':tracer_name,
                'classifier':classifier,
                "certainty_NID":certainty_NID,
                'imaging_modality':imaging_modality,
                'date_of_the_tracing':date_of_the_tracing,
                'soma_position':soma_position}

        swc = neuron.nodes
        swc['label'] = swc['label'].astype('int')
        swc.loc[swc['type'] == 'slab', 'label'] = 1
        swc.loc[swc['type'] == 'root', 'label'] = 1
        swc.loc[swc['type'] == 'branch', 'label'] = 5
        swc.loc[swc['type'] == 'end', 'label'] = 6
        swc.drop('type', axis=1, inplace=True)

        neuron = navis.read_swc(Path(rf"W:\Florian\function-neurotransmitter-morphology\{item['Volume']}\{item['Volume']}-000_registered.swc"),
                                id=str(id), name= name, units= units,
                                tracer_name= tracer_name, classifier= classifier,
                                certainty_NID= certainty_NID, imaging_modality= imaging_modality, date_of_the_tracing= date_of_the_tracing,soma_position= soma_position)


        swc = fix_duplicates(swc)

        neuron.nodes = swc
        neuron.soma = 1
        neuron.nodes.loc[:, 'radius'] = 0.3
        # neuron.nodes.loc[1:, 'radius'] = 0.3
        # neuron.nodes.iloc[0, 5] = 2
        neuron_mesh = navis.conversion.tree2meshneuron(neuron, use_normals=True, tube_points=20)


        neuron.to_swc(Path(rf"W:\Florian\function-neurotransmitter-morphology\{item['Volume']}\{item['Volume']}-000_registered_metadata.swc"),write_meta=my_meta)
        neuron.to_swc(Path(rf"W:\Florian\function-neurotransmitter-morphology\all_swc\{item['Volume']}-000_registered_metadata.swc"), write_meta=my_meta)
        navis.write_mesh(neuron_mesh, rf"W:\Florian\function-neurotransmitter-morphology\all_obj\{item['Volume']}-000_registered_metadata.obj")
        navis.write_mesh(neuron_mesh, rf"W:\Florian\function-neurotransmitter-morphology\{item['Volume']}\{item['Volume']}-000_registered_metadata.obj")
    #find the rows that actually have ROIs
    for i,item in cell_register.iterrows():
        valid_files = ''
        if ' ' in item['Function'].rstrip():
            for functional_name in item.Function.split(' '):
                temp_path = fr'W:\Florian\function-neurotransmitter-morphology\functional\{functional_name}\{functional_name}_roi.tiff'
                if Path(temp_path).exists() and valid_files == '':
                    valid_files += functional_name
                elif Path(temp_path).exists():
                    valid_files = valid_files + " " + functional_name
            cell_register.loc[cell_register['Internal name']==item["Internal name"], 'Functional'] = valid_files
        else:
            temp_path = fr'W:\Florian\function-neurotransmitter-morphology\functional\{item.Function}\{item.Function}_roi.tiff'
            if Path(temp_path).exists():
                valid_files += item.Function

        cell_register.loc[cell_register['Internal name'] == item["Internal name"], 'Functional'] = valid_files


    cell_register = cell_register[cell_register['Functional'].isna()|~(cell_register['Functional']=='')]    #drop empty rows







    cell_register['Volume'] = cell_register['Volume'].apply(check_registered_swc)

    cell_register =  cell_register[~cell_register['Volume'].isna()]


    #write metadata



    volumes = [navis.read_mesh(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Retina.obj',
                                  units='microns',output='volume'),
                  navis.read_mesh(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Midbrain.obj',
                                  units='microns',output='volume'),
                  navis.read_mesh(
                      r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Forebrain.obj',
                      units='microns',output='volume'),
                  navis.read_mesh(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Habenula.obj',
                                  units='microns',output='volume'),
                  navis.read_mesh(
                      r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Hindbrain.obj',
                      units='microns',output='volume'),
                  navis.read_mesh(
                      r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_mece\Spinal Cord.obj',
                      units='microns',output='volume'),
                  navis.read_mesh(
                      r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Gad1b Cluster 1.obj',
                      units='microns',output='volume'),
                  navis.read_mesh(
                      r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Gad1b Cluster 2.obj',
                      units='microns',output='volume'),
                  navis.read_mesh(
                      r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Vglut2 cluster 1.obj',
                      units='microns',output='volume'),
                  navis.read_mesh(
                      Path(r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Vglut2 cluster 2.obj'),
                      units='microns',output='volume'),
                  navis.read_mesh(
                      r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Raphe - Superior.obj',
                      units='microns',output='volume'),
                  navis.read_mesh(
                      r'Y:\Zebrafish atlases\z_brain_atlas\region_masks_z_brain_1_0\Rhombencephalon - Neuropil Region 6.obj',
                      units='microns',output='volume'),
                  ]







