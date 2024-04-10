import numpy as np
import navis
import pandas as pd
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
def load_mesh(cell,path,swc=False):
    if swc:
        if cell['imaging_modality'] == 'clem':

            cell_name = 'clem_zfish1_' + cell.type + '_' + str(cell.cell_name)
            clem_path = path.joinpath('clem_zfish1').joinpath('all_cells').joinpath(cell_name).joinpath('mapped')


            if clem_path.joinpath(cell_name + '_mapped.swc').exists():
                cell['swc'] = navis.read_swc(clem_path.joinpath(cell_name + '_mapped.swc'), units="um")
            else:
                print(f"No cell found at {clem_path.joinpath(cell_name + '_mapped.swc')}")
                cell['swc'] = np.nan
            return cell

        if cell['imaging_modality'] == 'photoactivation':
            pa_path = path.joinpath('paGFP').joinpath(cell.cell_name)

            if pa_path.joinpath(cell.cell_name + '.swc').exists():
                cell['swc'] = navis.read_swc(pa_path.joinpath(cell.cell_name + '.swc'), units="um")
            else:
                cell['swc'] = np.nan
            return cell
        
    else:
        if cell['imaging_modality'] == 'clem':
    
            cell_name = 'clem_zfish1_' + cell.type + '_'+ str(cell.cell_name)
            clem_path = path.joinpath('clem_zfish1').joinpath('all_cells').joinpath(cell_name).joinpath('mapped')
    
            
            if clem_path.joinpath(cell_name + '_axon_mapped.obj').exists():
                cell['axon_mesh'] = navis.read_mesh(clem_path.joinpath(cell_name  + '_axon_mapped.obj'),units="um")
            else:
                print(f"No axon found at {clem_path.joinpath(cell_name  + '_axon_mapped.obj')}")
                cell['axon_mesh'] = np.nan
    
            if clem_path.joinpath(cell_name  + '_dendrite_mapped.obj').exists():
                cell['dendrite_mesh'] = navis.read_mesh(clem_path.joinpath(cell_name  + '_dendrite_mapped.obj'),units="um")
            else:
                print(f"No dendrite found at {clem_path.joinpath(cell_name  + '_dendrite_mapped.obj')}")
                cell['dendrite_mesh'] = np.nan
    
            if clem_path.joinpath(cell_name  + '_soma_mapped.obj').exists():
                cell['soma_mesh'] = navis.read_mesh(clem_path.joinpath(cell_name + '_soma_mapped.obj'),units="um")
            else:
                print(f"No soma found at {clem_path.joinpath(cell_name  + '_soma_mapped.obj')}")
                cell['soma_mesh'] = np.nan
            return cell
    
        if cell['imaging_modality'] == 'photoactivation':
            pa_path = path.joinpath('paGFP').joinpath(cell.cell_name)
    
    
            if pa_path.joinpath(cell.cell_name + '.obj').exists():
                cell['neurites_mesh'] = navis.read_mesh(pa_path.joinpath(cell.cell_name + '.obj'),units="um")
            else:
                cell['neurites_mesh'] = np.nan
    
    
            if pa_path.joinpath(cell.cell_name + '_soma.obj').exists():
                cell['soma_mesh'] = navis.read_mesh(pa_path.joinpath(cell.cell_name + '_soma.obj'),units="um")
            else:
                cell['soma_mesh'] = np.nan
            return cell

