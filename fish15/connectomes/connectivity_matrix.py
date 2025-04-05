import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# Constants
PATH_ALL_CELLS = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224.xlsx'
ROOT_FOLDER = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/traced_neurons/all_cells_111224/'
OUTPUT_CSV = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224_with_hemisphere.csv'
OUTPUT_PATH = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/connectomes/connectivity_matrices'
LDA_CSV_PATH = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224_with_hemisphere_lda_lof_022825.csv'

# Define the color dictionary
COLOR_CELL_TYPE_DICT = {
    "integrator_ipsilateral": (254/255, 179/255, 38/255, 0.7),      # Yellow-orange
    "integrator_contralateral": (232/255, 77/255, 138/255, 0.7),    # Magenta-pink
    "dynamic_threshold": (100/255, 197/255, 235/255, 0.7),          # Light blue
    "motor_command": (127/255, 88/255, 175/255, 0.7),               # Purple
    "myelinated": (80/255, 220/255, 100/255, 0.7),                  # Dark gray for axons
    "axon_rostral": (105/255, 105/255, 105/255, 0.7),  # Dim Gray (Darker)
    "axon_caudal": (192/255, 192/255, 192/255, 0.7),   # Silver (Lighter)
}

COLOR_CELL_TYPE_DICT_LR = {
    # Integrator ipsilateral
    "integrator_ipsilateral_left": (254/255, 179/255, 38/255, 0.7),   # Yellow-orange
    "integrator_ipsilateral_right": (254/255, 179/255, 38/255, 0.7),  # Yellow-orange

    # Integrator contralateral
    "integrator_contralateral_left": (232/255, 77/255, 138/255, 0.7), # Magenta-pink
    "integrator_contralateral_right": (232/255, 77/255, 138/255, 0.7),# Magenta-pink

    # Dynamic threshold
    "dynamic_threshold_left": (100/255, 197/255, 235/255, 0.7),       # Light blue
    "dynamic_threshold_right": (100/255, 197/255, 235/255, 0.7),      # Light blue

    # Motor command
    "motor_command_left": (127/255, 88/255, 175/255, 0.7),            # Purple
    "motor_command_right": (127/255, 88/255, 175/255, 0.7),           # Purple

    # Myelinated
    "myelinated_left": (80/255, 220/255, 100/255, 0.7),               # Green
    "myelinated_right": (80/255, 220/255, 100/255, 0.7),              # Green

    # Axon rostral
    "axon_rostral_left": (105/255, 105/255, 105/255, 0.7),            # Gray
    "axon_rostral_right": (105/255, 105/255, 105/255, 0.7),           # Gray

    # Axon caudal
    "axon_caudal_left": (192/255, 192/255, 192/255, 0.7),             # Light gray
    "axon_caudal_right": (192/255, 192/255, 192/255, 0.7),            # Light gray
}

# Functions
def get_inputs_outputs_by_hemisphere_general(root_folder, seed_cell_ids, hemisphere_df):
    """
    Extract and categorize input/output neurons for given seed cell IDs based on hemisphere.
    Results include same-side and different-side synapses and cells for inputs and outputs,
    along with calculated percentages for each category.
    """
    # Load hemisphere data w/out duplicates 
    hemisphere_df['nucleus_id'] = hemisphere_df['nucleus_id'].astype(str)
    # Replace nucleus_id with axon_id where nucleus_id == '0'
    hemisphere_df.loc[hemisphere_df['nucleus_id'] == '0', 'nucleus_id'] = hemisphere_df['axon_id'].astype(str)
    hemisphere_map = hemisphere_df.set_index('nucleus_id')['hemisphere'].to_dict()

    # Initialize results
    results = {
        "outputs": {
            "cells": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "synapses": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "percentages": {"synapses": 0.0}
        },
        "inputs": {
            "cells": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "synapses": {"same_side": pd.DataFrame(), "different_side": pd.DataFrame()},
            "percentages": {"synapses": 0.0}
        },
        "counters": {"output_seed_counter": 0, "input_seed_counter": 0}
    }

    for seed_cell_id in seed_cell_ids:
        # Determine hemisphere of the seed cell
        seed_cell_hemisphere = hemisphere_map.get(str(seed_cell_id), None)
        if seed_cell_hemisphere is None:
            print(f"Seed cell ID {seed_cell_id} has no hemisphere data. Skipping.")
            continue

        #### OUTPUTS ####
        # Find the presynaptic (output) file
        output_file_pattern = f"clem_zfish1_cell_{seed_cell_id}_ng_res_presynapses.csv"
        output_file_path = None

        # Search for the cell presynaptic file
        for root, _, files in os.walk(root_folder):
            for filename in fnmatch.filter(files, output_file_pattern):
                output_file_path = os.path.join(root, filename)
                break

        # If the cell presynaptic file is not found, fall back to the axon presynaptic file
        if not output_file_path or not os.path.exists(output_file_path):
            output_file_pattern = f"clem_zfish1_axon_{seed_cell_id}_ng_res_presynapses.csv"
            for root, _, files in os.walk(root_folder):
                for filename in fnmatch.filter(files, output_file_pattern):
                    output_file_path = os.path.join(root, filename)
                    break

        if output_file_path:
            # Load and process outputs
            outputs_data = pd.read_csv(output_file_path, comment='#', sep=' ', header=None,
                                       names=["partner_cell_id", "x", "y", "z", "synapse_id", "size",
                                              "prediction_status", "validation_status", "date"])
            valid_outputs = outputs_data[outputs_data['validation_status'].str.contains('valid', na=False)]
            output_ids = valid_outputs['partner_cell_id']

            traced_dendrites = output_ids[output_ids.isin(hemisphere_df['dendrite_id'])]
            matched_outputs = [
                hemisphere_df[hemisphere_df['dendrite_id'] == dendrite].iloc[0]
                for dendrite in traced_dendrites
            ] if not traced_dendrites.empty else []

            output_connected_cells = pd.DataFrame(matched_outputs)
            if not output_connected_cells.empty:
                output_connected_cells_unique = output_connected_cells.drop_duplicates(subset='axon_id')

                # Calculate percentages
                output_percentage_synapses = len(output_connected_cells) / len(valid_outputs) if len(valid_outputs) > 0 else 0
                results["outputs"]["percentages"]["synapses"] += output_percentage_synapses

                results["counters"]["output_seed_counter"] += 1

                # Categorize by hemisphere
                if 'hemisphere' in output_connected_cells_unique.columns:
                    same_outputs_cells = output_connected_cells_unique[output_connected_cells_unique['hemisphere'] == seed_cell_hemisphere]
                    different_outputs_cells = output_connected_cells_unique[output_connected_cells_unique['hemisphere'] != seed_cell_hemisphere]

                    same_outputs_synapses = output_connected_cells[output_connected_cells['hemisphere'] == seed_cell_hemisphere]
                    different_outputs_synapses = output_connected_cells[output_connected_cells['hemisphere'] != seed_cell_hemisphere]
                else:
                    same_outputs_cells = pd.DataFrame()
                    different_outputs_cells = pd.DataFrame()
                    same_outputs_synapses = pd.DataFrame()
                    different_outputs_synapses = pd.DataFrame()

                # Fill NaN with 'not functionally imaged' to handle missing classifier values 
                # Usefull when only looking at functionally imaged neurons 
                dataframes = [same_outputs_cells, different_outputs_cells, same_outputs_synapses, different_outputs_synapses]
                for i, df in enumerate(dataframes):
                    if 'functional classifier' in df.columns:
                        # Ensure we're working with a copy and modifying it in place
                        df = df.copy()
                        
                        # Explicitly cast and fill NaN values
                        df['functional classifier'] = df['functional classifier'].astype('object')
                        df.loc[:, 'functional classifier'] = df['functional classifier'].fillna('not functionally imaged')
                        
                        # Update the original DataFrame
                        dataframes[i] = df

                # Reassign modified DataFrames back to their original variables
                same_outputs_cells, different_outputs_cells, same_outputs_synapses, different_outputs_synapses = dataframes

                # Append to results
                results["outputs"]["cells"]["same_side"] = pd.concat(
                    [results["outputs"]["cells"]["same_side"], same_outputs_cells], ignore_index=True)
                results["outputs"]["cells"]["different_side"] = pd.concat(
                    [results["outputs"]["cells"]["different_side"], different_outputs_cells], ignore_index=True)
                results["outputs"]["synapses"]["same_side"] = pd.concat(
                    [results["outputs"]["synapses"]["same_side"], same_outputs_synapses], ignore_index=True)
                results["outputs"]["synapses"]["different_side"] = pd.concat(
                    [results["outputs"]["synapses"]["different_side"], different_outputs_synapses], ignore_index=True)

        #### INPUTS ####
        # Find the postsynaptic (input) file
        input_file_pattern = f"clem_zfish1_cell_{seed_cell_id}_ng_res_postsynapses.csv"
        input_file_path = None

        for root, _, files in os.walk(root_folder):
            for filename in fnmatch.filter(files, input_file_pattern):
                input_file_path = os.path.join(root, filename)
                break

        if input_file_path:
            # Load and process inputs
            inputs_data = pd.read_csv(input_file_path, comment='#', sep=' ', header=None,
                                      names=["partner_cell_id", "x", "y", "z", "synapse_id", "size",
                                             "prediction_status", "validation_status", "date"])
            valid_inputs = inputs_data[inputs_data['validation_status'].str.contains('valid', na=False)]
            input_ids = valid_inputs['partner_cell_id']

            traced_axons = input_ids[input_ids.isin(hemisphere_df['axon_id'])]
            matched_inputs = [
                hemisphere_df[hemisphere_df['axon_id'] == axon].iloc[0]
                for axon in traced_axons
            ] if not traced_axons.empty else []

            input_connected_cells = pd.DataFrame(matched_inputs)
            if not input_connected_cells.empty:
                input_connected_cells_unique = input_connected_cells.drop_duplicates(subset='axon_id')

                # Calculate percentages
                input_percentage_synapses = len(input_connected_cells) / len(valid_inputs) if len(valid_inputs) > 0 else 0
                results["inputs"]["percentages"]["synapses"] += input_percentage_synapses

                results["counters"]["input_seed_counter"] += 1

                # Categorize by hemisphere
                if 'hemisphere' in input_connected_cells_unique.columns:
                    same_inputs_cells = input_connected_cells_unique[input_connected_cells_unique['hemisphere'] == seed_cell_hemisphere]
                    different_inputs_cells = input_connected_cells_unique[input_connected_cells_unique['hemisphere'] != seed_cell_hemisphere]

                    same_inputs_synapses = input_connected_cells[input_connected_cells['hemisphere'] == seed_cell_hemisphere]
                    different_inputs_synapses = input_connected_cells[input_connected_cells['hemisphere'] != seed_cell_hemisphere]
                else:
                    same_inputs_cells = pd.DataFrame()
                    different_inputs_cells = pd.DataFrame()
                    same_inputs_synapses = pd.DataFrame()
                    different_inputs_synapses = pd.DataFrame()

                # Fill NaN with 'not functionally imaged' to handle missing classifier values
                # Usefull when only looking at functionally imaged neurons 
                dataframes = [same_inputs_cells, different_inputs_cells, same_inputs_synapses, different_inputs_synapses]
                for i, df in enumerate(dataframes):
                    if 'functional classifier' in df.columns:
                        # Ensure we're working with a copy and modifying it in place
                        df = df.copy()

                        # Explicitly cast and fill NaN values
                        df['functional classifier'] = df['functional classifier'].astype('object')
                        df.loc[:, 'functional classifier'] = df['functional classifier'].fillna('not functionally imaged')

                        # Update the original DataFrame
                        dataframes[i] = df

                # Reassign modified DataFrames back to their original variables
                same_inputs_cells, different_inputs_cells, same_inputs_synapses, different_inputs_synapses = dataframes

                # Append to results
                results["inputs"]["cells"]["same_side"] = pd.concat(
                    [results["inputs"]["cells"]["same_side"], same_inputs_cells], ignore_index=True)
                results["inputs"]["cells"]["different_side"] = pd.concat(
                    [results["inputs"]["cells"]["different_side"], different_inputs_cells], ignore_index=True)
                results["inputs"]["synapses"]["same_side"] = pd.concat(
                    [results["inputs"]["synapses"]["same_side"], same_inputs_synapses], ignore_index=True)
                results["inputs"]["synapses"]["different_side"] = pd.concat(
                    [results["inputs"]["synapses"]["different_side"], different_inputs_synapses], ignore_index=True)

    return results

def generate_directional_connectivity_matrix_general(root_folder, seg_ids, df_w_hemisphere):
    """
    Generate a directional connectivity matrix for functionally imaged neurons.

    Inputs:
        - root_folder: Path to the folder with neuron connectivity data.
        - df_w_hemisphere: DataFrame containing neuron metadata with hemisphere data.
    
    Output:
        - A directional connectivity matrix (inputs vs outputs) with functional IDs as labels.
    """

    # Initialize directional matrix with functional IDs as labels
    seg_ids = [str(id) for id in seg_ids]
    connectivity_matrix = pd.DataFrame(0, index=seg_ids, columns=seg_ids)

    # Initialize a set to store globally counted non-zero synapse IDs
    stored_nonzero_synapse_ids = set()

    for source_id in seg_ids:
        # Fetch the connectivity data for the source neuron
        results = get_inputs_outputs_by_hemisphere_general(root_folder, [source_id], df_w_hemisphere)

        # Try to fetch the output synapse table for the cell
        cell_file_name = f"clem_zfish1_cell_{source_id}_ng_res_presynapses.csv"
        cell_name = f"clem_zfish1_cell_{source_id}"
        cell_output_file_path = os.path.join(root_folder, cell_name, cell_file_name)

        if os.path.exists(cell_output_file_path):
            output_file_path = cell_output_file_path
        else: 
            # If the cell file doesn't exist, fall back to the axon synapse table
            axon_file_name = f"clem_zfish1_axon_{source_id}_ng_res_presynapses.csv"
            axon_name = f"clem_zfish1_axon_{source_id}"
            output_file_path = os.path.join(root_folder, axon_name, axon_file_name)
            
        outputs_data = pd.read_csv(output_file_path, comment='#', sep=' ', header=None,
                                   names=["partner_cell_id", "x", "y", "z", "synapse_id", "size",
                                          "prediction_status", "validation_status", "date"])
        valid_outputs = outputs_data[outputs_data['validation_status'].str.contains('valid', na=False)]

        # Process OUTPUT connections
        for direction in ["same_side", "different_side"]:
            outputs = results["outputs"]["synapses"][direction]
            if not outputs.empty and "nucleus_id" in outputs.columns:
                # Ensure seg_ids is not empty
                if len(seg_ids) > 0:  # Check that seg_ids is not empty
                    try:
                        # Convert both outputs["nucleus_id"] and seg_ids to integers
                        outputs["nucleus_id"] = outputs["nucleus_id"].astype(int)
                        seg_ids = [int(id) for id in seg_ids]

                        # Filter the outputs DataFrame
                        outputs = outputs[outputs["nucleus_id"].isin(seg_ids)]
                    except ValueError as e:
                        print(f"Error converting nucleus_id or seg_ids to integers: {e}")
                else:
                    print("Warning: seg_ids is empty. No filtering applied.")

                for _, output_row in outputs.iterrows():
                    # Isolate corresponding synapses
                    target_dendrite = output_row["dendrite_id"]
                    matching_row = valid_outputs[valid_outputs["partner_cell_id"] == target_dendrite]
                    synapse_ids = matching_row['synapse_id'].tolist()

                    # Count the number of synapse_id == 0
                    zero_synapse_count = synapse_ids.count(0)

                    # Filter non-zero synapse IDs
                    nonzero_synapse_ids = [sid for sid in synapse_ids if sid != 0]

                    # Identify new non-zero synapses not already counted
                    new_nonzero_synapses = [sid for sid in nonzero_synapse_ids if sid not in stored_nonzero_synapse_ids]

                    # Calculate the total number of new synapses
                    num_new_synapses = zero_synapse_count + len(new_nonzero_synapses)

                    # Update the connectivity matrix if there are new synapses
                    if num_new_synapses > 0:
                        target_func_id = str(output_row["nucleus_id"])
                        connectivity_matrix.loc[source_id, target_func_id] += num_new_synapses  # Update OUTPUTS ONLY

                        # Add new non-zero synapses to the globally stored set
                        stored_nonzero_synapse_ids.update(new_nonzero_synapses)

        # Process INPUT connections
        # Fetch the input synapse table
        file_name = f"clem_zfish1_cell_{source_id}_ng_res_postsynapses.csv"
        input_file_path = os.path.join(root_folder, cell_name, file_name)

        # Check if the file exists before processing (it won't exist for an axon)
        if os.path.exists(input_file_path):
            
            inputs_data = pd.read_csv(input_file_path, comment='#', sep=' ', header=None,
                                    names=["partner_cell_id", "x", "y", "z", "synapse_id", "size",
                                            "prediction_status", "validation_status", "date"])
            valid_inputs = inputs_data[inputs_data['validation_status'].str.contains('valid', na=False)]

            for direction in ["same_side", "different_side"]:
                inputs = results["inputs"]["synapses"][direction]
                if not inputs.empty and "nucleus_id" in inputs.columns:
                    
                    # Ensure seg_ids is not empty
                    if len(seg_ids) > 0:  # Check that seg_ids is not empty
                        try:
                            # Convert both outputs["nucleus_id"] and seg_ids to integers
                            inputs["nucleus_id"] = inputs["nucleus_id"].astype(int)
                            seg_ids = [int(id) for id in seg_ids]

                            # Filter the outputs DataFrame
                            inputs = inputs[inputs["nucleus_id"].isin(seg_ids)]
                        except ValueError as e:
                            print(f"Error converting nucleus_id or seg_ids to integers: {e}")
                    else:
                        print("Warning: seg_ids is empty. No filtering applied.")

                    for _, input_row in inputs.iterrows():
                        # Isolate corresponding synapses
                        target_axon = input_row["axon_id"]
                        matching_row = valid_inputs[valid_inputs["partner_cell_id"] == target_axon]
                        synapse_ids = matching_row['synapse_id'].tolist()

                        # Count the number of synapse_id == 0
                        zero_synapse_count = synapse_ids.count(0)

                        # Filter non-zero synapse IDs
                        nonzero_synapse_ids = [sid for sid in synapse_ids if sid != 0]

                        # Identify new non-zero synapses not already counted
                        new_nonzero_synapses = [sid for sid in nonzero_synapse_ids if sid not in stored_nonzero_synapse_ids]

                        # Calculate the total number of new synapses
                        num_new_synapses = zero_synapse_count + len(new_nonzero_synapses)

                        # Update the connectivity matrix if there are new synapses
                        if num_new_synapses > 0: 
                            source_input_func_id = str(input_row["nucleus_id"])
                            connectivity_matrix.loc[source_input_func_id, source_id] += num_new_synapses  # Update INPUTS ONLY

                            # Add new non-zero synapses to the globally stored set
                            stored_nonzero_synapse_ids.update(new_nonzero_synapses)

    return connectivity_matrix

def load_and_clean_data(path, drop_duplicates=True):
    """Load data and optionally drop duplicates by 'axon_id'"""
    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
    if drop_duplicates:
        df = df.drop_duplicates(subset='axon_id')
    return df

def standardize_naming(df):
    """Standardize naming in 'functional classifier' column."""
    replacements = {'dynamic threshold': 'dynamic_threshold', 'motor command': 'motor_command'}
    df['functional classifier'] = df['functional classifier'].replace(replacements)
    return df

def fetch_filtered_ids(df, col_1, condition_1, col_2=None, condition_2=None):
    """Filter DataFrame by conditions and return unique nucleus and functional IDs."""
    filtered = df[df.iloc[:, col_1] == condition_1]
    if col_2 and condition_2:
        filtered = filtered[filtered.iloc[:, col_2] == condition_2]
    return filtered.iloc[:, 5].drop_duplicates(), filtered.iloc[:, 1].drop_duplicates()

def create_nucleus_id_groups(df):
    """Group nucleus IDs by functional types."""
    groups = {
        "axon_rostral": df.loc[(df['type'] == 'axon') & (df['comment'] == 'axon exits the volume rostrally'), 'axon_id'],
        "integrator_ipsilateral": fetch_filtered_ids(df, 9, 'integrator', 11, 'ipsilateral')[0],
        "integrator_contralateral": fetch_filtered_ids(df, 9, 'integrator', 11, 'contralateral')[0],
        "dynamic_threshold": fetch_filtered_ids(df, 9, 'dynamic_threshold')[0],
        "motor_command": fetch_filtered_ids(df, 9, 'motor_command')[0],
        "myelinated": df.loc[(df['type'] == 'cell') & (df['functional classifier'] == 'myelinated'), 'nucleus_id'],
        "axon_caudal": df.loc[(df['type'] == 'axon') & (df['comment'] == 'axon exits the volume caudally'), 'axon_id'],
        #"nfi": df.loc[(df['type'] == 'cell') & (df['functional_id'] == 'not functionally imaged'), 'nucleus_id']
    }
    return {k: [str(id) for id in v] for k, v in groups.items()}

def generate_functional_types(nucleus_id_groups):
    """Create a dictionary mapping nucleus IDs to functional types."""
    return {nucleus_id: functional_type
            for functional_type, ids in nucleus_id_groups.items()
            for nucleus_id in ids}

def filter_connectivity_matrix(matrix, functional_types):
    """Filter the connectivity matrix and functional types by non-zero indices."""
    non_zero_indices = matrix.index[(matrix.sum(axis=1) != 0) | (matrix.sum(axis=0) != 0)]
    filtered_matrix = matrix.loc[non_zero_indices, non_zero_indices]
    filtered_types = {k: v for k, v in functional_types.items() if k in filtered_matrix.index}
    return filtered_matrix, filtered_types

def process_matrix(matrix, df):
    """
    Modify the matrix rows based on the 'neurotransmitter classifier' in df.

    Parameters:
    - matrix (pd.DataFrame): The matrix to be modified.
    - df (pd.DataFrame): DataFrame containing the 'neurotransmitter classifier' and 'nucleus_id' columns.
    
    Returns:
    - pd.DataFrame: Modified matrix.
    """
    # Iterate through each row in the matrix
    for idx in matrix.index:
        # Fetch the corresponding row in df where 'nucleus_id' matches the matrix index
        df_row = df.loc[df["nucleus_id"] == idx]

        # Ensure there is a match
        if df_row.empty:
            raise ValueError(f"Index '{idx}' in the matrix does not have a matching 'nucleus_id' in df.")

        # Extract the neurotransmitter classifier
        classifier = df_row.iloc[0]["neurotransmitter classifier"]

        if classifier == "inhibitory":
            # Multiply the entire row by -1
            matrix.loc[idx] *= -1
        #elif classifier == "unknown":
        #    # Replace non-zero values with NaN
        #    matrix.loc[idx] = matrix.loc[idx].apply(lambda x: np.nan if x != 0 else 0)
        # If classifier is 'excitatory', no change is needed

    return matrix

def plot_connectivity_matrix(matrix, functional_types, output_path, category_order, df=None, title="Directional Connectivity Matrix", display_type="normal", plot_type="heatmap", color_cell_type_dict=None):
    """
    Plots a connectivity matrix with support for differentiating inhibitory and excitatory synapses.

    Parameters:
    - matrix (pd.DataFrame): The connectivity matrix to plot.
    - functional_types (dict): Mapping of matrix indices to functional categories.
    - output_path (str): Directory to save the output plot.
    - category_order (list): Ordered list of functional categories for sorting and grouping.
    - df (pd.DataFrame): Optional DataFrame with 'nucleus_id' and 'neurotransmitter classifier' for preprocessing.
    - title (str): Title of the plot.
    - display_type (str): "normal" or "Inhibitory_Excitatory".
    - plot_type (str): "heatmap" or "scatter".
    - color_cell_type_dict (dict): Dictionary mapping functional types to RGBA color tuples.
    """
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # Set default COLOR_CELL_TYPE_DICT if not provided
    if color_cell_type_dict is None:
        color_cell_type_dict = {
            "axon_rostral": (0.8, 0.3, 0.3, 0.7),  # Example: Red for axon_rostral
            "integrator_ipsilateral": (0.3, 0.8, 0.3, 0.7),  # Green
            "integrator_contralateral": (0.3, 0.3, 0.8, 0.7),  # Blue
            "dynamic_threshold": (0.8, 0.8, 0.3, 0.7),  # Yellow
            "motor_command": (0.5, 0.3, 0.5, 0.7),  # Purple
            "axon_caudal": (0.3, 0.5, 0.8, 0.7),  # Cyan
        }

    # Validate input parameters
    if display_type not in ["normal", "Inhibitory_Excitatory"]:
        raise ValueError("Invalid display_type. Choose 'normal' or 'Inhibitory_Excitatory'.")
    if plot_type not in ["heatmap", "scatter"]:
        raise ValueError("Invalid plot_type. Choose 'heatmap' or 'scatter'.")

    # Pre-process the matrix for "Inhibitory_Excitatory" display
    if display_type == "Inhibitory_Excitatory":
        if df is not None:
            matrix = process_matrix(matrix, df)
        else:
            raise ValueError("DataFrame (df) is required for 'Inhibitory_Excitatory' display.")

    # Filter and sort the matrix for any display type
    functional_types = {k: v for k, v in functional_types.items() if v in category_order and k in matrix.index}

    # Filter matrix to include only rows/columns from CATEGORY_ORDER
    filtered_indices = [
        idx for idx in matrix.index if functional_types.get(idx, "unknown") in category_order
    ]
    filtered_matrix = matrix.loc[filtered_indices, filtered_indices]

    # Sort indices based on CATEGORY_ORDER
    def sort_key(func_id):
        category = functional_types.get(func_id, "unknown")
        return category_order.index(category) if category in category_order else len(category_order)

    sorted_indices = sorted(filtered_indices, key=sort_key)
    matrix_with_nan = filtered_matrix.loc[sorted_indices, sorted_indices].copy()

    # Handle "Inhibitory_Excitatory" clipping and colormap
    if display_type == "Inhibitory_Excitatory":
        matrix_with_nan = np.clip(matrix_with_nan, -2, 2)
        colors = [
            "#D62839",  # Strongly Inhibitory (Ruby Red)
            "#F88F54",  # Weakly Inhibitory (Peach Orange)
            "#FFFFFF",  # Zero (White)
            "#9FD598",  # Weakly Excitatory (Pale Jade Green)
            "#227C71"   # Strongly Excitatory (Teal Green)
        ]
        cmap = mcolors.ListedColormap(colors, name="Inhibitory-Excitatory")
        bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
    else:
        # Define default colormap and bounds for "normal"
        cmap = mcolors.ListedColormap(["white", "blue", "green", "yellow", "pink", "red"])
        bounds = [0, 1, 2, 3, 4, 5, 6]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))

    if plot_type == "heatmap":
        # Heatmap with matshow
        cax = ax.matshow(matrix_with_nan, cmap=cmap, norm=norm)
    elif plot_type == "scatter":
        # Scatter plot
        x, y = np.meshgrid(range(len(matrix_with_nan.columns)), range(len(matrix_with_nan.index)))
        x, y = x.flatten(), y.flatten()
        sizes = np.abs(matrix_with_nan.values.flatten()) * 100  # Scale dot sizes based on absolute matrix values
        colors = matrix_with_nan.values.flatten()
        scatter = ax.scatter(x, y, c=colors, s=sizes, cmap=cmap, norm=norm)

     # Set square aspect ratio
    ax.set_aspect("equal")

    # Add color bar
    cbar = plt.colorbar(
        cax if plot_type == "heatmap" else scatter,
        ax=ax,
        boundaries=bounds,
        ticks=[-2, -1, 0, 1, 2] if display_type == "Inhibitory_Excitatory" else [0, 1, 2, 3, 4, 5],
        spacing="uniform",
        orientation="horizontal",
        pad=0.1,
    )
    cbar_label = "Synapse Strength (Inhibitory to Excitatory)" if display_type == "Inhibitory_Excitatory" else "No. of Synapses"
    cbar.set_label(cbar_label)

    # Labels and title
    ax.set_xticks(range(len(matrix_with_nan.columns)))
    ax.set_yticks(range(len(matrix_with_nan.index)))
    ax.set_xticklabels(matrix_with_nan.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(matrix_with_nan.index, fontsize=8)
    ax.set_xlabel("Pre-synaptic (Axons)")
    ax.set_ylabel("Post-synaptic (Dendrites)")
    ax.set_title(title, fontsize=12)

    # Functional type bars
    bar_width = 3  # Adjust width of the side bars (for the left bar)
    bar_height = 3  # Adjust height of the top bars (for the top bar)

    for i, functional_id in enumerate(matrix_with_nan.index):
        if functional_id not in functional_types:
            print(f"Warning: Functional ID '{functional_id}' not found in functional_types.")
            continue

        functional_type = functional_types[functional_id]
        color = color_cell_type_dict.get(functional_type, (0.8, 0.8, 0.8, 0.7))  # Default to light gray

        # Draw the left bar (row indicators) with correct alignment
        ax.add_patch(
            patches.Rectangle((-bar_width, i - 0.5), bar_width, 1, color=color, zorder=2)
        )

        # Draw the top bar (column indicators) with correct alignment
        ax.add_patch(
            patches.Rectangle((i - 0.5, -bar_height), 1, bar_height, color=color, zorder=2)
        )

    # Add gridlines to separate groups
    group_boundaries = []
    last_type = None
    for i, idx in enumerate(matrix_with_nan.index):
        current_type = functional_types.get(idx, "unknown")
        if current_type != last_type:
            group_boundaries.append(i - 0.5)
            last_type = current_type
    group_boundaries.append(len(matrix_with_nan.index) - 0.5)

    for boundary in group_boundaries:
        ax.axhline(boundary, color="black", linewidth=1.5, zorder=3)
        ax.axvline(boundary, color="black", linewidth=1.5, zorder=3)

    # Adjust axis limits
    ax.set_xlim(-1.5, len(matrix_with_nan.columns) - 0.5)
    ax.set_ylim(len(matrix_with_nan.index) - 0.5, -1.5)

    # Adjust layout and save
    plt.tight_layout()
    sanitized_title = title.lower().replace(" ", "_").replace(":", "").replace("/", "_")
    output_pdf_path = os.path.join(output_path, f"{sanitized_title}.pdf")
    plt.savefig(output_pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()

# Main Workflow
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Matrix without enhancement  

# Step 1: Load and preprocess data
all_cells = load_and_clean_data(OUTPUT_CSV)
all_cells = standardize_naming(all_cells)

# Step 2: Generate nucleus ID groups and functional types
nucleus_id_groups = create_nucleus_id_groups(all_cells)
functional_types = generate_functional_types(nucleus_id_groups)

# Step 3: Generate connectivity matrix
all_ids_nuc = np.concatenate([v for v in nucleus_id_groups.values()])
connectivity_matrix = generate_directional_connectivity_matrix_general(ROOT_FOLDER, all_ids_nuc, all_cells)

# Step 4: Filter and plot connectivity matrix
filtered_matrix, filtered_types = filter_connectivity_matrix(connectivity_matrix, functional_types)
category_order = ["axon_rostral", "integrator_ipsilateral", "integrator_contralateral", "dynamic_threshold", "motor_command", "myelinated", "axon_caudal"]

plot_connectivity_matrix(
    filtered_matrix, 
    filtered_types, 
    OUTPUT_PATH, 
    category_order, 
    df=all_cells,
    title="non_enhanced_wnfi", display_type="normal", plot_type="heatmap",
    color_cell_type_dict = COLOR_CELL_TYPE_DICT,
)

# Statistics for the text. 
# Count elements in each category
counts = {key: len(value) for key, value in filtered_types.items()}

# Print the counts
for category, count in counts.items():
    print(f"{category}: {count}")

# All reconstructed neurons
# 301 - 84 (FI) - 9 (neg controls) - 42 (RS, not connected). 

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Matrix L/R split 

def create_nucleus_id_groups_hemisphere(df):
    """Group nucleus IDs by functional types and hemispheres."""
    groups = {
        # Axon rostral, split by hemisphere
        "axon_rostral_left": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume rostrally') &
            (df['hemisphere'] == 'L'),
            'axon_id'
        ],
        "axon_rostral_right": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume rostrally') &
            (df['hemisphere'] == 'R'),
            'axon_id'
        ],
        # Integrators split by hemisphere
        "integrator_ipsilateral_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'ipsilateral') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "integrator_ipsilateral_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'ipsilateral') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        "integrator_contralateral_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'contralateral') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "integrator_contralateral_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'contralateral') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Dynamic threshold, split by hemisphere
        "dynamic_threshold_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'dynamic_threshold') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "dynamic_threshold_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'dynamic_threshold') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Motor command, split by hemisphere
        "motor_command_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'motor_command') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "motor_command_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'motor_command') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Myelinated, split by hemisphere
        "myelinated_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'myelinated') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "myelinated_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'myelinated') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Axon caudal, split by hemisphere
        "axon_caudal_left": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume caudally') &
            (df['hemisphere'] == 'L'),
            'axon_id'
        ],
        "axon_caudal_right": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume caudally') &
            (df['hemisphere'] == 'R'),
            'axon_id'
        ]
    }
    # Convert all IDs to strings for consistency
    return {k: [str(id) for id in v] for k, v in groups.items()}

# Step 1: Load and preprocess data
all_cells = load_and_clean_data(OUTPUT_CSV)
all_cells = standardize_naming(all_cells)

# Step 2: Generate nucleus ID groups and functional types
nucleus_id_groups = create_nucleus_id_groups_hemisphere(all_cells)
functional_types = generate_functional_types(nucleus_id_groups)

# Step 3: Generate connectivity matrix
all_ids_nuc = np.concatenate([v for v in nucleus_id_groups.values()])
connectivity_matrix = generate_directional_connectivity_matrix_general(ROOT_FOLDER, all_ids_nuc, all_cells)

# Step 4: Filter and plot connectivity matrix
filtered_matrix, filtered_types = filter_connectivity_matrix(connectivity_matrix, functional_types)

CATEGORY_ORDER = [
    "axon_rostral_left",
    "integrator_ipsilateral_left", "integrator_contralateral_left",
    "dynamic_threshold_left", "motor_command_left", "myelinated_left", 
    "axon_caudal_left",
    "axon_rostral_right",
    "integrator_ipsilateral_right", "integrator_contralateral_right",
    "dynamic_threshold_right", "motor_command_right", "myelinated_right",
    "axon_caudal_right",
]

CATEGORY_ORDER_short = [
    "integrator_ipsilateral_left", "integrator_contralateral_left",
    "dynamic_threshold_left", "motor_command_left", "myelinated_left", 
    "integrator_ipsilateral_right", "integrator_contralateral_right",
    "dynamic_threshold_right", "motor_command_right", "myelinated_right",
]

plot_connectivity_matrix(
    filtered_matrix, 
    filtered_types, 
    OUTPUT_PATH, 
    CATEGORY_ORDER, 
    df=all_cells,
    title="non_enhanced_lr", display_type="Inhibitory_Excitatory", plot_type="scatter",
    color_cell_type_dict = COLOR_CELL_TYPE_DICT_LR,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Matrix with enhancement  

# Step 1: Load and preprocess data
all_cells = load_and_clean_data(LDA_CSV_PATH)
all_cells = standardize_naming(all_cells)

# Step 2: Generate nucleus ID groups and functional types
nucleus_id_groups = create_nucleus_id_groups(all_cells)
functional_types = generate_functional_types(nucleus_id_groups)

# Step 3: Generate connectivity matrix
all_ids_nuc = np.concatenate([v for v in nucleus_id_groups.values()])
connectivity_matrix = generate_directional_connectivity_matrix_general(ROOT_FOLDER, all_ids_nuc, all_cells)

# Step 4: Filter and plot connectivity matrix
filtered_matrix, filtered_types = filter_connectivity_matrix(connectivity_matrix, functional_types)
category_order = ["axon_rostral", "integrator_ipsilateral", "integrator_contralateral", "dynamic_threshold", "motor_command", "axon_caudal"]

plot_connectivity_matrix(
    filtered_matrix, 
    filtered_types, 
    OUTPUT_PATH, 
    category_order, 
    df=all_cells,
    title="Normal Heatmap", display_type="Inhibitory_Excitatory", plot_type="scatter",
    color_cell_type_dict = COLOR_CELL_TYPE_DICT,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Matrix L/R split with enhancement
def create_nucleus_id_groups_hemisphere(df):
    """Group nucleus IDs by functional types and hemispheres."""
    groups = {
        # Axon rostral, split by hemisphere
        "axon_rostral_left": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume rostrally') &
            (df['hemisphere'] == 'L'),
            'axon_id'
        ],
        "axon_rostral_right": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume rostrally') &
            (df['hemisphere'] == 'R'),
            'axon_id'
        ],
        # Integrators split by hemisphere
        "integrator_ipsilateral_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'ipsilateral') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "integrator_ipsilateral_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'ipsilateral') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        "integrator_contralateral_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'contralateral') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "integrator_contralateral_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'contralateral') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Dynamic threshold, split by hemisphere
        "dynamic_threshold_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'dynamic_threshold') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "dynamic_threshold_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'dynamic_threshold') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Motor command, split by hemisphere
        "motor_command_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'motor_command') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "motor_command_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'motor_command') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Myelinated, split by hemisphere
        "myelinated_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'myelinated') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "myelinated_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'myelinated') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Axon caudal, split by hemisphere
        "axon_caudal_left": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume caudally') &
            (df['hemisphere'] == 'L'),
            'axon_id'
        ],
        "axon_caudal_right": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume caudally') &
            (df['hemisphere'] == 'R'),
            'axon_id'
        ]
    }
    # Convert all IDs to strings for consistency
    return {k: [str(id) for id in v] for k, v in groups.items()}

# Step 1: Load and preprocess data
all_cells = load_and_clean_data(LDA_CSV_PATH)
all_cells = standardize_naming(all_cells)

# Step 2: Generate nucleus ID groups and functional types
nucleus_id_groups = create_nucleus_id_groups_hemisphere(all_cells)
functional_types = generate_functional_types(nucleus_id_groups)

# Step 3: Generate connectivity matrix
all_ids_nuc = np.concatenate([v for v in nucleus_id_groups.values()])
connectivity_matrix = generate_directional_connectivity_matrix_general(ROOT_FOLDER, all_ids_nuc, all_cells)

# Step 4: Filter and plot connectivity matrix
filtered_matrix, filtered_types = filter_connectivity_matrix(connectivity_matrix, functional_types)

CATEGORY_ORDER = [
    "axon_rostral_left",
    "integrator_ipsilateral_left", "integrator_contralateral_left",
    "dynamic_threshold_left", "motor_command_left", "myelinated_left", 
    "axon_caudal_left", 
    "axon_rostral_right",
    "integrator_ipsilateral_right", "integrator_contralateral_right",
    "dynamic_threshold_right", "motor_command_right", "myelinated_right",
    "axon_caudal_right",
]

CATEGORY_ORDER_short = [
    "integrator_ipsilateral_left", "integrator_contralateral_left",
    "dynamic_threshold_left", "motor_command_left", "myelinated_left", 
    "integrator_ipsilateral_right", "integrator_contralateral_right",
    "dynamic_threshold_right", "motor_command_right", "myelinated_right",
]

plot_connectivity_matrix(
    filtered_matrix, 
    filtered_types, 
    OUTPUT_PATH, 
    CATEGORY_ORDER_short, 
    df=all_cells,
    title="enhanced_lr_short", display_type="Inhibitory_Excitatory", plot_type="scatter",
    color_cell_type_dict = COLOR_CELL_TYPE_DICT_LR,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Matrix L/R split with enhancement and predicted below native

def create_nucleus_id_groups_hemisphere(df):
    """Group nucleus IDs by functional types, hemispheres, and lda categories."""
    groups = {
        # Axon rostral, split by hemisphere
        "axon_rostral_left": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume rostrally') &
            (df['hemisphere'] == 'L'),
            'axon_id'
        ],
        "axon_rostral_right": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume rostrally') &
            (df['hemisphere'] == 'R'),
            'axon_id'
        ],
        # Integrators ipsi split by hemisphere and lda
        "integrator_ipsilateral_left_native": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'ipsilateral') &
            (df['hemisphere'] == 'L') &
            (df['lda'] == 'native'),
            'nucleus_id'
        ],
        "integrator_ipsilateral_left_predicted": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'ipsilateral') &
            (df['hemisphere'] == 'L') &
            (df['lda'] == 'predicted'),
            'nucleus_id'
        ],
        "integrator_ipsilateral_right_native": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'ipsilateral') &
            (df['hemisphere'] == 'R') &
            (df['lda'] == 'native'),
            'nucleus_id'
        ],
        "integrator_ipsilateral_right_predicted": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'ipsilateral') &
            (df['hemisphere'] == 'R') &
            (df['lda'] == 'predicted'),
            'nucleus_id'
        ],
        # Integrators contra split by hemisphere and lda
        "integrator_contralateral_left_native": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'contralateral') &
            (df['hemisphere'] == 'L') &
            (df['lda'] == 'native'),
            'nucleus_id'
        ],
        "integrator_contralateral_left_predicted": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'contralateral') &
            (df['hemisphere'] == 'L') &
            (df['lda'] == 'predicted'),
            'nucleus_id'
        ],
        "integrator_contralateral_right_native": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'contralateral') &
            (df['hemisphere'] == 'R') &
            (df['lda'] == 'native'),
            'nucleus_id'
        ],
        "integrator_contralateral_right_predicted": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'integrator') &
            (df['projection classifier'] == 'contralateral') &
            (df['hemisphere'] == 'R') &
            (df['lda'] == 'predicted'),
            'nucleus_id'
        ],
        # Dynamic threshold, split by hemisphere and lda
        "dynamic_threshold_left_native": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'dynamic_threshold') &
            (df['hemisphere'] == 'L') &
            (df['lda'] == 'native'),
            'nucleus_id'
        ],
        "dynamic_threshold_left_predicted": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'dynamic_threshold') &
            (df['hemisphere'] == 'L') &
            (df['lda'] == 'predicted'),
            'nucleus_id'
        ],
        "dynamic_threshold_right_native": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'dynamic_threshold') &
            (df['hemisphere'] == 'R') &
            (df['lda'] == 'native'),
            'nucleus_id'
        ],
        "dynamic_threshold_right_predicted": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'dynamic_threshold') &
            (df['hemisphere'] == 'R') &
            (df['lda'] == 'predicted'),
            'nucleus_id'
        ],
        # Motor command, split by hemisphere and lda
        "motor_command_left_native": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'motor_command') &
            (df['hemisphere'] == 'L') &
            (df['lda'] == 'native'),
            'nucleus_id'
        ],
        "motor_command_left_predicted": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'motor_command') &
            (df['hemisphere'] == 'L') &
            (df['lda'] == 'predicted'),
            'nucleus_id'
        ],
        "motor_command_right_native": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'motor_command') &
            (df['hemisphere'] == 'R') &
            (df['lda'] == 'native'),
            'nucleus_id'
        ],
        "motor_command_right_predicted": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'motor_command') &
            (df['hemisphere'] == 'R') &
            (df['lda'] == 'predicted'),
            'nucleus_id'
        ],
        # Myelinated, split by hemisphere
        "myelinated_left": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'myelinated') &
            (df['hemisphere'] == 'L'),
            'nucleus_id'
        ],
        "myelinated_right": df.loc[
            (df['type'] == 'cell') &
            (df['functional classifier'] == 'myelinated') &
            (df['hemisphere'] == 'R'),
            'nucleus_id'
        ],
        # Axon caudal, split by hemisphere
        "axon_caudal_left": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume caudally') &
            (df['hemisphere'] == 'L'),
            'axon_id'
        ],
        "axon_caudal_right": df.loc[
            (df['type'] == 'axon') &
            (df['comment'] == 'axon exits the volume caudally') &
            (df['hemisphere'] == 'R'),
            'axon_id'
        ]
    }
    # Convert all IDs to strings for consistency
    return {k: [str(id) for id in v] for k, v in groups.items()}

COLOR_CELL_TYPE_DICT_LR_prednat = {
    # Integrator ipsilateral
    "integrator_ipsilateral_left_native": (254/255, 179/255, 38/255, 0.7),   # Yellow-orange
    "integrator_ipsilateral_left_predicted": (254/255, 220/255, 80/255, 0.7),# Lighter Yellow-orange
    "integrator_ipsilateral_right_native": (254/255, 179/255, 38/255, 0.7),  # Yellow-orange
    "integrator_ipsilateral_right_predicted": (254/255, 220/255, 80/255, 0.7),# Lighter Yellow-orange

    # Integrator contralateral
    "integrator_contralateral_left_native": (232/255, 77/255, 138/255, 0.7), # Magenta-pink
    "integrator_contralateral_left_predicted": (255/255, 105/255, 180/255, 0.7),# Lighter Magenta-pink
    "integrator_contralateral_right_native": (232/255, 77/255, 138/255, 0.7),# Magenta-pink
    "integrator_contralateral_right_predicted": (255/255, 105/255, 180/255, 0.7),# Lighter Magenta-pink

    # Dynamic threshold
    "dynamic_threshold_left_native": (100/255, 197/255, 235/255, 0.7),       # Light blue
    "dynamic_threshold_left_predicted": (173/255, 216/255, 230/255, 0.7),   # Lighter Light blue
    "dynamic_threshold_right_native": (100/255, 197/255, 235/255, 0.7),      # Light blue
    "dynamic_threshold_right_predicted": (173/255, 216/255, 230/255, 0.7),   # Lighter Light blue

    # Motor command
    "motor_command_left_native": (127/255, 88/255, 175/255, 0.7),            # Purple
    "motor_command_left_predicted": (186/255, 104/255, 200/255, 0.7),       # Lighter Purple
    "motor_command_right_native": (127/255, 88/255, 175/255, 0.7),           # Purple
    "motor_command_right_predicted": (186/255, 104/255, 200/255, 0.7),      # Lighter Purple

    # Myelinated
    "myelinated_left": (80/255, 220/255, 100/255, 0.7),               # Green
    "myelinated_right": (80/255, 220/255, 100/255, 0.7),              # Green

    # Axon rostral
    "axon_rostral_left": (105/255, 105/255, 105/255, 0.7),            # Gray
    "axon_rostral_right": (105/255, 105/255, 105/255, 0.7),           # Gray

    # Axon caudal
    "axon_caudal_left": (192/255, 192/255, 192/255, 0.7),             # Light gray
    "axon_caudal_right": (192/255, 192/255, 192/255, 0.7),            # Light gray
}

# Step 1: Load and preprocess data
all_cells = load_and_clean_data(LDA_CSV_PATH)
all_cells = standardize_naming(all_cells)

# Step 2: Generate nucleus ID groups and functional types
nucleus_id_groups = create_nucleus_id_groups_hemisphere(all_cells)
functional_types = generate_functional_types(nucleus_id_groups)

# Step 3: Generate connectivity matrix
all_ids_nuc = np.concatenate([v for v in nucleus_id_groups.values()])
connectivity_matrix = generate_directional_connectivity_matrix_general(ROOT_FOLDER, all_ids_nuc, all_cells)

# Step 4: Filter and plot connectivity matrix
filtered_matrix, filtered_types = filter_connectivity_matrix(connectivity_matrix, functional_types)

CATEGORY_ORDER = [
    "axon_rostral_left",
    "integrator_ipsilateral_left_native", "integrator_ipsilateral_left_predicted",
    "integrator_contralateral_left_native", "integrator_contralateral_left_predicted",
    "dynamic_threshold_left_native", "dynamic_threshold_left_predicted",
    "motor_command_left_native", "motor_command_left_predicted",
    "myelinated_left",
    "axon_caudal_left",
    "axon_rostral_right",
    "integrator_ipsilateral_right_native", "integrator_ipsilateral_right_predicted",
    "integrator_contralateral_right_native", "integrator_contralateral_right_predicted",
    "dynamic_threshold_right_native", "dynamic_threshold_right_predicted",
    "motor_command_right_native", "motor_command_right_predicted",
    "myelinated_right",
    "axon_caudal_right",
]

CATEGORY_ORDER_short = [
    "integrator_ipsilateral_left_native", "integrator_ipsilateral_left_predicted",
    "integrator_contralateral_left_native", "integrator_contralateral_left_predicted",
    "dynamic_threshold_left_native", "dynamic_threshold_left_predicted",
    "motor_command_left_native", "motor_command_left_predicted",
    "myelinated_left",
    "integrator_ipsilateral_right_native", "integrator_ipsilateral_right_predicted",
    "integrator_contralateral_right_native", "integrator_contralateral_right_predicted",
    "dynamic_threshold_right_native", "dynamic_threshold_right_predicted",
    "motor_command_right_native", "motor_command_right_predicted",
    "myelinated_right",
]

plot_connectivity_matrix(
    filtered_matrix, 
    filtered_types, 
    OUTPUT_PATH, 
    CATEGORY_ORDER, 
    df=all_cells,
    title="enhanced_lr_pred_split", display_type="Inhibitory_Excitatory", plot_type="scatter",
    color_cell_type_dict = COLOR_CELL_TYPE_DICT_LR_prednat,
)

# %%
