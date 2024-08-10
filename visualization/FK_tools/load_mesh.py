import numpy as np
import navis
import pandas as pd
from pathlib import Path
import math
import warnings

import numpy as np
import navis
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

def load_mesh(cell, path, swc=False, use_smooth_pa=False, load_both=False,load_repaired=False):
    if cell['imaging_modality'] == 'clem' and not load_repaired:
        cell_name_clem = f'clem_zfish1_{cell.cell_name}'
        clem_path = path / 'clem_zfish1' / 'all_cells' / cell_name_clem / 'mapped'
        clem_path_to_predict = path / 'clem_zfish1' / 'to_predict' / cell_name_clem / 'mapped'
        clem_path_neg_controls = path / 'clem_zfish1' / 'neg_controls' / cell_name_clem / 'mapped'
    elif cell['imaging_modality'] == 'clem' and load_repaired:
        cell_name_clem = f'clem_zfish1_{cell.cell_name}'
        clem_path = path / 'clem_zfish1' / 'all_cells_repaired'
        clem_path_to_predict = path / 'clem_zfish1' / 'to_predict' / cell_name_clem / 'mapped'
        regular_clem_path = path / 'clem_zfish1' / 'all_cells' / cell_name_clem / 'mapped'
        clem_path_neg_controls = path / 'clem_zfish1' / 'neg_controls' / cell_name_clem / 'mapped'



    elif cell['imaging_modality'] == 'EM' and not load_repaired:
        cell_name_em = f'em_fish1_{cell.cell_name}'
        cell_name_em_obj = f'em_fish1_{cell.cell_name}'
        path_seed_cells_em = path / 'em_zfish1' / 'data_seed_cells' / 'output_data' / cell_name_em / 'mapped'
        path_89189_postsynaptic_em = path / 'em_zfish1' / 'data_cell_89189_postsynaptic_partners' / 'output_data' / cell_name_em / 'mapped'
        path_10_postsynaptic_em = path / 'em_zfish1' / 'cell_010_postsynaptic_partners' / 'output_data' / cell_name_em / 'mapped'
        path_11_postsynaptic_em = path / 'em_zfish1' / 'cell_011_postsynaptic_partners' / 'output_data' / cell_name_em / 'mapped'
        path_19_postsynaptic_em = path / 'em_zfish1' / 'cell_019_postsynaptic_partners' / 'output_data' / cell_name_em / 'mapped'
        cell_name_em_obj = f'em_fish1_{cell.cell_name}'
        path_seed_cells_em_obj = path / 'em_zfish1' / 'data_seed_cells' / 'output_data' / cell_name_em_obj / 'mapped'
        path_89189_postsynaptic_em_obj = path / 'em_zfish1' / 'data_cell_89189_postsynaptic_partners' / 'output_data' / cell_name_em_obj / 'mapped'
        path_10_postsynaptic_em_obj = path / 'em_zfish1' / 'cell_010_postsynaptic_partners' / 'output_data' / cell_name_em_obj / 'mapped'
        path_11_postsynaptic_em_obj = path / 'em_zfish1' / 'cell_011_postsynaptic_partners' / 'output_data' / cell_name_em_obj / 'mapped'
        path_19_postsynaptic_em_obj = path / 'em_zfish1' / 'cell_019_postsynaptic_partners' / 'output_data' / cell_name_em_obj / 'mapped'
    elif cell['imaging_modality'] == 'EM' and load_repaired:
        cell_name_em = f'em_zfish1_{cell.cell_name}'
        path_seed_cells_em =            path / 'em_zfish1' / 'all_cells_repaired'
        path_89189_postsynaptic_em =    path / 'em_zfish1' / 'all_cells_repaired'
        path_10_postsynaptic_em =       path / 'em_zfish1' / 'all_cells_repaired'
        path_11_postsynaptic_em =       path / 'em_zfish1' / 'all_cells_repaired'
        path_19_postsynaptic_em =       path / 'em_zfish1' / 'all_cells_repaired'
        cell_name_em_obj = f'em_fish1_{cell.cell_name}'
        path_seed_cells_em_obj = path / 'em_zfish1' / 'data_seed_cells' / 'output_data' / cell_name_em_obj / 'mapped'
        path_89189_postsynaptic_em_obj = path / 'em_zfish1' / 'data_cell_89189_postsynaptic_partners' / 'output_data' / cell_name_em_obj / 'mapped'
        path_10_postsynaptic_em_obj = path / 'em_zfish1' / 'cell_010_postsynaptic_partners' / 'output_data' / cell_name_em_obj / 'mapped'
        path_11_postsynaptic_em_obj = path / 'em_zfish1' / 'cell_011_postsynaptic_partners' / 'output_data' / cell_name_em_obj / 'mapped'
        path_19_postsynaptic_em_obj = path / 'em_zfish1' / 'cell_019_postsynaptic_partners' / 'output_data' / cell_name_em_obj / 'mapped'





    elif cell['imaging_modality'] == 'photoactivation':
        pa_path = path / 'paGFP' / str(cell.cell_name)

    if load_both:
        swc = True

    def load_file(file_path, file_type, is_swc=False):
        if file_path.exists():

                return navis.read_swc(file_path, units="um", read_meta=False) if is_swc else navis.read_mesh(file_path, units="um")
        else:
            print(f"No {file_type} found at {file_path}")
            return np.nan

    if swc:
        file_suffix = '_smoothed.swc' if use_smooth_pa else '.swc'
        if cell['imaging_modality'] == 'photoactivation':
            file_path = pa_path / f'{cell.cell_name}{file_suffix}'
            cell['swc'] = load_file(file_path, 'SWC', is_swc=True)
        elif cell['imaging_modality'] == 'clem'  and not load_repaired:
            if clem_path.exists():
                file_path = clem_path / f'{cell_name_clem}_mapped.swc'
            elif clem_path_to_predict.exists():
                file_path = clem_path_to_predict / f'{cell_name_clem}_mapped.swc'
            else:
                file_path = clem_path_neg_controls / f'{cell_name_clem}_mapped.swc'

            cell['swc'] = load_file(file_path, 'SWC', is_swc=True)


        elif cell['imaging_modality'] == 'clem'  and  load_repaired:
            if clem_path.exists():
                file_path = clem_path / f'{cell_name_clem}_repaired.swc'
            elif clem_path_to_predict.exists():
                file_path = clem_path_to_predict / f'{cell_name_clem}_mapped.swc'

            else:
                file_path = clem_path_neg_controls / f'{cell_name_clem}_mapped.swc'
            cell['swc'] = load_file(file_path, 'SWC', is_swc=True)


        elif cell['imaging_modality'] == 'EM':
                if load_repaired:
                    suffix = '_repaired'
                else:
                    suffix = '_mapped'

                if path_seed_cells_em.exists():
                    file_path = path_seed_cells_em / f'{cell_name_em}{suffix}.swc'

                    cell['swc'] = load_file(file_path, 'SWC', is_swc=True)
                elif path_89189_postsynaptic_em.exists():
                    file_path = path_89189_postsynaptic_em / f'{cell_name_em}{suffix}.swc'
                    cell['swc'] = load_file(file_path, 'SWC', is_swc=True)
                elif path_10_postsynaptic_em.exists():
                    file_path = path_10_postsynaptic_em / f'{cell_name_em}{suffix}.swc'
                    cell['swc'] = load_file(file_path, 'SWC', is_swc=True)
                elif path_11_postsynaptic_em.exists():
                    file_path = path_11_postsynaptic_em / f'{cell_name_em}{suffix}.swc'
                    cell['swc'] = load_file(file_path, 'SWC', is_swc=True)
                elif path_19_postsynaptic_em.exists():
                    file_path = path_19_postsynaptic_em / f'{cell_name_em}{suffix}.swc'
                    cell['swc'] = load_file(file_path, 'SWC', is_swc=True)



    if not swc or load_both:
        if cell['imaging_modality'] == 'clem':
            if (clem_path / f'{cell_name_clem}').exists():
                file_path = clem_path / f'{cell_name_clem}'
            elif (clem_path_to_predict / f'{cell_name_clem}').exists():
                file_path = clem_path_to_predict / f'{cell_name_clem}'
            elif clem_path_neg_controls.exists():
                file_path = clem_path_neg_controls / f'{cell_name_clem}_mapped.swc'

            else:
                file_path = regular_clem_path /  f'{cell_name_clem}'


            for component,element_type in zip(['_axon_mapped','_dendrite_mapped','_soma_mapped'],['axon','dendrite','soma']):
                try:
                    cell[f'{element_type}_mesh'] = load_file(Path(str(file_path) + f"{component}.obj"), element_type)
                except:
                    pass
            try:
                cell['all_mesh'] = load_file(Path(str(file_path) + "_mapped.obj"), 'complete')
            except:
                pass
        elif cell['imaging_modality'] == 'EM':
            if path_seed_cells_em_obj.exists():
                file_path = path_seed_cells_em_obj
            elif path_89189_postsynaptic_em_obj.exists():
                file_path = path_89189_postsynaptic_em_obj
            elif path_10_postsynaptic_em_obj.exists():
                file_path = path_10_postsynaptic_em_obj
            elif path_11_postsynaptic_em_obj.exists():
                file_path = path_11_postsynaptic_em_obj
            elif path_19_postsynaptic_em_obj.exists():
                file_path = path_19_postsynaptic_em_obj



            cell['axon_mesh'] = load_file(file_path / f'{cell_name_em_obj}_axon_mapped.obj', 'axon')
            cell['dendrite_mesh'] = load_file(file_path / f'{cell_name_em_obj}_dendrite_mapped.obj', 'dendrite')
            cell['soma_mesh'] = load_file(file_path / f'{cell_name_em_obj}_soma_mapped.obj', 'soma')
        elif cell['imaging_modality'] == 'photoactivation':
            file_suffix = '_smoothed.obj' if use_smooth_pa else '.obj'
            cell['neurites_mesh'] = load_file(pa_path / f'{cell.cell_name}{file_suffix}', 'neurites')
            cell['soma_mesh'] = load_file(pa_path / f'{cell.cell_name}_soma.obj', 'soma')
            cell['all_mesh'] = load_file(pa_path / f'{cell.cell_name}_combined.obj', 'Combined file')

    if type(cell['swc']) != float:
        cell['swc'].nodes.loc[:, 'radius'] = 0.5
        cell['swc'].nodes.loc[0, 'radius'] = 2
    if type(cell['swc']) == float:
        pass
    return cell
