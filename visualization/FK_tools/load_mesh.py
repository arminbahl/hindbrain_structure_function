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

def load_mesh(cell, path, swc=False, use_smooth_pa=False, load_both=False):
    if cell['imaging_modality'] == 'clem':
        cell_name_clem = f'clem_zfish1_{cell.type}_{cell.cell_name}'
        clem_path = path / 'clem_zfish1' / 'all_cells' / cell_name_clem / 'mapped'
    elif cell['imaging_modality'] == 'EM':
        cell_name_em = f'em_fish1_{cell.cell_name}'

        path_seed_cells_em = path / 'em_zfish1' / 'data_seed_cells' / 'output_data' / cell_name_em / 'mapped'
        path_89189_postsynaptic_em = path / 'em_zfish1' / 'data_cell_89189_postsynaptic_partners' / 'output_data' / cell_name_em / 'mapped'
        path_10_postsynaptic_em = path / 'em_zfish1' / 'cell_010_postsynaptic_partners' / 'output_data' / cell_name_em / 'mapped'
        path_11_postsynaptic_em = path / 'em_zfish1' / 'cell_011_postsynaptic_partners' / 'output_data' / cell_name_em / 'mapped'
        path_19_postsynaptic_em = path / 'em_zfish1' / 'cell_019_postsynaptic_partners' / 'output_data' / cell_name_em / 'mapped'
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
        elif cell['imaging_modality'] == 'clem':
            file_path = clem_path / f'{cell_name_clem}_mapped.swc'
            cell['swc'] = load_file(file_path, 'SWC', is_swc=True)
        elif cell['imaging_modality'] == 'EM':
                if path_seed_cells_em.exists():
                    file_path = path_seed_cells_em / f'{cell_name_em}_mapped.swc'

                    cell['swc'] = load_file(file_path, 'SWC', is_swc=True)
                elif path_89189_postsynaptic_em.exists():
                    file_path = path_89189_postsynaptic_em / f'{cell_name_em}_mapped.swc'
                    cell['swc'] = load_file(file_path, 'SWC', is_swc=True)
                elif path_10_postsynaptic_em.exists():
                    file_path = path_10_postsynaptic_em / f'{cell_name_em}_mapped.swc'
                    cell['swc'] = load_file(file_path, 'SWC', is_swc=True)
                elif path_11_postsynaptic_em.exists():
                    file_path = path_11_postsynaptic_em / f'{cell_name_em}_mapped.swc'
                    cell['swc'] = load_file(file_path, 'SWC', is_swc=True)
                elif path_19_postsynaptic_em.exists():
                    file_path = path_19_postsynaptic_em / f'{cell_name_em}_mapped.swc'
                    cell['swc'] = load_file(file_path, 'SWC', is_swc=True)


    if not swc or load_both:
        if cell['imaging_modality'] == 'clem':
            cell['axon_mesh'] = load_file(clem_path / f'{cell_name_clem}_axon_mapped.obj', 'axon')
            cell['dendrite_mesh'] = load_file(clem_path / f'{cell_name_clem}_dendrite_mapped.obj', 'dendrite')
            cell['soma_mesh'] = load_file(clem_path / f'{cell_name_clem}_soma_mapped.obj', 'soma')
            cell['all_mesh'] = load_file(clem_path / f'{cell_name_clem}_mapped.obj', 'soma')
        elif cell['imaging_modality'] == 'EM':
            if path_seed_cells_em.exists():
                file_path = path_seed_cells_em
            elif path_89189_postsynaptic_em.exists():
                file_path = path_89189_postsynaptic_em
            elif path_10_postsynaptic_em.exists():
                file_path = path_10_postsynaptic_em
            elif path_11_postsynaptic_em.exists():
                file_path = path_11_postsynaptic_em
            elif path_19_postsynaptic_em.exists():
                file_path = path_19_postsynaptic_em



            cell['axon_mesh'] = load_file(file_path / f'{cell_name_em}_axon_mapped.obj', 'axon')
            cell['dendrite_mesh'] = load_file(file_path / f'{cell_name_em}_dendrite_mapped.obj', 'dendrite')
            cell['soma_mesh'] = load_file(file_path / f'{cell_name_em}_soma_mapped.obj', 'soma')
        elif cell['imaging_modality'] == 'photoactivation':
            file_suffix = '_smoothed.obj' if use_smooth_pa else '.obj'
            cell['neurites_mesh'] = load_file(pa_path / f'{cell.cell_name}{file_suffix}', 'neurites')
            cell['soma_mesh'] = load_file(pa_path / f'{cell.cell_name}_soma.obj', 'soma')
            cell['all_mesh'] = load_file(pa_path / f'{cell_name}_combined.obj', 'Combined file')

    return cell
