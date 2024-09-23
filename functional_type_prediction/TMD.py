import tmd
import navis
import pandas as pd
from tmd.utils import SOMA_TYPE
from tmd.Soma import Soma
from tmd.utils import TREE_TYPE_DICT
import os
from tmd.io.h5 import read_h5
from tmd.utils import TmdError
import warnings
import numpy as np
from tmd.io.swc import SWC_DCT
from scipy import sparse as sp
from scipy.sparse import csgraph as cs
if __name__ == "__main__":
    class LoadNeuronError(TmdError):
        """Captures the exception of failing to load a single neuron."""
    def redefine_types(user_types=None):
        """Return tree types depending on the customized types selected by the user.

        Args:
            user_types (dictionary or None):

        Returns:
            final_types (dict): tree types for the construction of Neuron.
        """
        final_tree_types = TREE_TYPE_DICT.copy()
        if user_types is not None:
            final_tree_types.update(user_types)
        return final_tree_types
    def own_load_neuron(input_file, line_delimiter="\n", soma_type=None, user_tree_types=None, remove_duplicates=True,load_array=None):

        """I/O method to load an swc or h5 file into a Neuron object."""
        tree_types = redefine_types(user_tree_types)

        # Definition of swc types from type_dict function
        if soma_type is None:
            soma_index = SOMA_TYPE
        else:
            soma_index = soma_type

        # Make neuron with correct filename and load data
        ext = os.path.splitext(input_file)[-1].lower()
        if ext == ".swc":
            data = tmd.io.swc_to_data(tmd.io.read_swc(input_file=input_file, line_delimiter=line_delimiter))
            neuron = tmd.io.Neuron.Neuron(name=str(input_file).replace(".swc", ""))

        elif ext == ".h5":
            data = read_h5(input_file=input_file, remove_duplicates=remove_duplicates)
            neuron = tmd.io.Neuron.Neuron(name=str(input_file).replace(".h5", ""))

        else:
            raise LoadNeuronError(
                f"{input_file} is not a valid h5 or swc file. If asc set use_morphio to True."
            )

        # Check for duplicated IDs
        IDs, counts = np.unique(data[:, 0], return_counts=True)
        if (counts != 1).any():
            warnings.warn(f"The following IDs are duplicated: {IDs[counts > 1]}")

        data_T = np.transpose(data)

        try:
            soma_ids = np.where(data_T[1] == soma_index)[0]
        except IndexError as exc:
            raise LoadNeuronError("Soma points not in the expected format") from exc

        # Extract soma information from swc
        soma = Soma.Soma(
            x=data_T[SWC_DCT["x"]][soma_ids],
            y=data_T[SWC_DCT["y"]][soma_ids],
            z=data_T[SWC_DCT["z"]][soma_ids],
            d=data_T[SWC_DCT["radius"]][soma_ids],
        )

        # Save soma in Neuron
        neuron.set_soma(soma)
        p = np.array(data_T[6], dtype=int) - np.transpose(data)[0][0]
        # return p, soma_ids
        try:
            dA = sp.csr_matrix(
                (np.ones(len(p) - len(soma_ids)), (range(len(soma_ids), len(p)), p[len(soma_ids):])),
                shape=(len(p), len(p)),
            )
        except Exception as exc:
            raise LoadNeuronError("Cannot create connectivity, nodes not connected correctly.") from exc

        # assuming soma points are in the beginning of the file.
        comp = cs.connected_components(dA[len(soma_ids):, len(soma_ids):])

        # Extract trees
        for i in range(comp[0]):
            tree = tmd.io.make_tree(data[np.where(comp[1] == i)[0] + len(soma_ids)])
            neuron.append_tree(tree, tree_types)

        return neuron


    # New segment: TMD aka Armins paper https://link.springer.com/article/10.1007/s12021-017-9341-1
    path = r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\paGFP\20230413.1\20230413.1.swc"
    path = r"C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\clem_zfish1\all_cells_repaired\clem_zfish1_cell_576460752488813678_repaired.swc"



    neuron = own_load_neuron(path)
    neuron = neuron.simplify()
    pd = tmd.methods.get_ph_neuron(neuron.neurites[0])




    data = tmd.io.swc_to_data(tmd.io.read_swc(input_file=path, line_delimiter='\n'))
    neuron = tmd.Neuron.Neuron(name=str(path).replace(".swc", ""))

    data_T = np.transpose(data)
