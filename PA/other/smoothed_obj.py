import navis
import pandas as pd
from pathlib import Path
import numpy as np

def check_registered_swc(file_name):#drop rows where swc file is not registered
    if Path(rf"W:\Florian\function-neurotransmitter-morphology\{file_name}\{file_name}-000_registered.swc").exists():
        return file_name
    else:
        return np.nan

#prepare the cell register
cell_register = pd.read_csv(r"C:\Users\ag-bahl\Downloads\Table1 (1).csv")   #load data
cell_register = cell_register[~cell_register['Function'].isna()]    #drop empty rows

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


#drop cells that are not registered
cell_register['Volume'] = cell_register['Volume'].apply(check_registered_swc)
cell_register =  cell_register[~cell_register['Volume'].isna()]


#loop through neurons

smooth_window = 4

for i,item in cell_register.iterrows():
    neuron = navis.read_swc(Path(rf"W:\Florian\function-neurotransmitter-morphology\{item['Volume']}\{item['Volume']}-000_registered.swc"))
    neuron.soma = 1
    neuron.nodes.iloc[0, 5] = 2
    neuron.nodes.loc[:, 'radius'] = 0.3
    navis.smooth_skeleton(neuron,window=smooth_window,inplace=True)
    neuron_mesh = navis.conversion.tree2meshneuron(neuron, use_normals=True, tube_points=20)
    navis.write_mesh(neuron_mesh,rf"W:\Florian\function-neurotransmitter-morphology\all_obj\smoothed\{item['Volume']}_mapped_smooth_window{smooth_window}.obj")

