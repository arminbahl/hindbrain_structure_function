import pandas as pd
import numpy as np
from pathlib import Path
import navis
import os
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from datetime import datetime
from scipy.signal import savgol_filter

# Define the color dictionary
COLOR_CELL_TYPE_DICT = {
    "integrator_ipsilateral": (254/255, 179/255, 38/255, 0.7),      # Yellow-orange
    "integrator_contralateral": (232/255, 77/255, 138/255, 0.7),    # Magenta-pink
    "dynamic_threshold": (100/255, 197/255, 235/255, 0.7),          # Light blue
    "motor_command": (127/255, 88/255, 175/255, 0.7),               # Purple
    "myelinated": (80/255, 220/255, 100/255, 0.1),                  # Bright teal 
    "axon": (0.2, 0.2, 0.2, 0.7),                                   # Dark gray for axons
    "not functionally imaged": (0.5, 0.5, 0.5, 0.7),                # Gray for not functionally imaged
}

def determine_hemisphere(row, root_folder, width_brain=495.56, progress=None):
    """
    Determine the hemisphere of a neuron based on its mesh file, with a progress counter.

    Parameters:
    - row: pandas DataFrame row.
    - root_folder: Root folder containing traced neuron data.
    - width_brain: Width of the brain for determining hemispheres.
    - progress: A dictionary with keys 'processed_count' and 'total_rows' for tracking progress.

    Returns:
    - 'R' or 'L' for right or left hemisphere, or None in case of errors.
    """
    try:
        if row['type'] == 'cell':
            mesh_path = Path(root_folder) / f"clem_zfish1_cell_{row['nucleus_id']}" / 'mapped' / f"clem_zfish1_cell_{row['nucleus_id']}_mapped.obj"
        elif row['type'] == 'axon':
            mesh_path = Path(root_folder) / f"clem_zfish1_axon_{row['axon_id']}" / 'mapped' / f"clem_zfish1_axon_{row['axon_id']}_axon_mapped.obj"
        else:
            return None

        # Read the mesh
        mesh = navis.read_mesh(mesh_path, units="um")

        # Determine hemisphere
        result = 'R' if np.mean(mesh._vertices[:, 0]) > (width_brain / 2) else 'L'
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        result = None

    # Update and print progress
    if progress is not None:
        progress['processed_count'] += 1
        print(f"Processed {progress['processed_count']}/{progress['total_rows']} rows", end='\r')
    return result


def fetch_filtered_ids(df, col_1_index, condition_1, col_2_index=None, condition_2=None):
    """
    Fetch unique values from two specific columns based on one or two conditions.
    
    Returns:
        - Unique values from column 5 (e.g., nucleus IDs).
        - Unique values from column 1 (e.g., functional IDs).
    """
    # Apply the first condition
    filtered_rows = df.loc[df.iloc[:, col_1_index] == condition_1]
    
    # Apply the second condition if provided
    if col_2_index is not None and condition_2 is not None:
        filtered_rows = filtered_rows.loc[filtered_rows.iloc[:, col_2_index] == condition_2]
    
    # Extract unique values from columns 5 and 2
    nuclei_ids = filtered_rows.iloc[:, 5].drop_duplicates()
    functional_ids = filtered_rows.iloc[:, 1].drop_duplicates()
    
    return nuclei_ids, functional_ids


def get_inputs_outputs_by_hemisphere(root_folder, seed_cell_ids, hemisphere_df):
    """
    Extract and categorize input/output neurons for given seed cell IDs based on hemisphere.
    Results include same-side and different-side synapses and cells for inputs and outputs,
    along with calculated percentages for each category.
    """
    # Load hemisphere data w/out duplicates 
    hemisphere_df['nucleus_id'] = hemisphere_df['nucleus_id'].astype(str)
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


def compute_count_probabilities_from_results(results, functional_only=False):
    """
    Compute connection counts and probabilities for cells and synapses across 
    same-side and different-side hemispheres for both inputs and outputs.

    Parameters:
    - results: Dictionary containing categorized results (e.g., from `get_inputs_outputs_by_hemisphere`).
    - functional_only: Boolean flag to use only functionally recorded neurons.

    Returns:
    - A dictionary containing counts and probabilities for each category of inputs and outputs.
    """
    def _process_category(df, functional_only):
        """
        Compute counts and probabilities for a single category (cells or synapses).

        Parameters:
        - df: DataFrame containing the connection data.
        - functional_only: Boolean flag to exclude non-functional neurons.

        Returns:
        - A DataFrame with counts and probabilities for each unique connection type.
        """
        if df.empty:
            return pd.DataFrame()

        connections = []

        for _, row in df.iterrows():
            if row['type'] == 'axon':
                # Handle axonal cases
                axon_exit_direction = row.get('comment', None)
                if pd.notna(axon_exit_direction):
                    connections.append({
                        'Functional Classifier': 'axon',
                        'Neurotransmitter Classifier': None,
                        'Projection Classifier': None,
                        'Axon Exit Direction': axon_exit_direction,
                    })
            elif row['type'] == 'cell':
                # Handle neuronal cases
                if functional_only and row['functional_id'] == 'not functionally imaged' and row['functional classifier'] != 'myelinated':
                    continue  # Skip non-functional neurons if functional_only is True
                connections.append({
                    'Functional Classifier': row['functional classifier'],
                    'Neurotransmitter Classifier': row['neurotransmitter classifier'],
                    'Projection Classifier': row['projection classifier'],
                    'Axon Exit Direction': None,
                })

        if not connections:
            return pd.DataFrame()

        # Create a DataFrame, compute counts and probabilities
        connections_df = pd.DataFrame(connections).fillna('None')
        counts_df = connections_df.value_counts().reset_index(name='Count')
        counts_df['Probability'] = counts_df['Count'] / counts_df['Count'].sum()

        return counts_df

    # Initialize the results dictionary
    final_results = {
        conn_type: {
            side: {data_type: pd.DataFrame() for data_type in ["cells", "synapses"]}
            for side in ["same_side", "different_side"]
        }
        for conn_type in ["outputs", "inputs"]
    }

    # Iterate through inputs and outputs, process each category
    for conn_type in ["outputs", "inputs"]:
        for side in ["same_side", "different_side"]:
            for data_type in ["cells", "synapses"]:
                df = results.get(conn_type, {}).get(data_type, {}).get(side, pd.DataFrame())
                final_results[conn_type][side][data_type] = _process_category(df, functional_only)

    return final_results


def draw_two_layer_neural_net(ax, left, right, bottom, top, data_df, node_radius=0.015, 
                              input_circle_color='gray', input_cell_type='excitatory',
                              show_midline=True, proportional_lines=True, a=5, b=2,
                              connection_type='outputs', add_legend=True):
    """
    Draw a 2-layer neural network with 1 input node and output nodes based on the rows in the DataFrame.
    Differentiates circles by neurotransmitter classifier with thicker outlines and adds a legend.
    Uses Arial font for all text.

    :param ax: The matplotlib axis to draw the network on.
    :param left: Left boundary of the diagram.
    :param right: Right boundary of the diagram.
    :param bottom: Bottom boundary of the diagram.
    :param top: Top boundary of the diagram.
    :param data_df: Pandas DataFrame with columns 'Functional Classifier', 'Projection Classifier', 'Neurotransmitter Classifier',
                    'Probability', 'Count'.
    :param node_radius: Radius of the nodes.
    :param input_circle_color: Color of the circle in the first layer.
    :param input_cell_type: Type of the input cell ('excitatory' or 'inhibitory').
    :param show_midline: Whether to display a midline in the plot.
    :param proportional_lines: Whether the connection lines should be proportional to probabilities.
    :param a: Scaling factor for line/arrow/T-bar sizes (slope of the linear equation).
    :param b: Minimum size for line/arrow/T-bar sizes (intercept of the linear equation).
    :param connection_type: 'outputs' (default) or 'inputs', determines arrow/T-bar direction.
    :param add_legend: Whether to add a legend for the neurotransmitter classifier.
    """
    import numpy as np
    from matplotlib.patches import Circle, Patch
    from matplotlib.lines import Line2D
    from colorsys import rgb_to_hls, hls_to_rgb

    # Adjust plot width if midline is not shown
    if not show_midline:
        midpoint = (left + right) / 2
        if connection_type == 'outputs':
            right = midpoint + (right - midpoint) * 0.8  # Shrink right side
        elif connection_type == 'inputs':
            left = midpoint - (midpoint - left) * 0.8  # Shrink left side

    # Color dictionary
    COLOR_CELL_TYPE_DICT = {
        "integrator_ipsilateral": (254/255, 179/255, 38/255, 0.7),      # Yellow-orange
        "integrator_contralateral": (232/255, 77/255, 138/255, 0.7),    # Magenta-pink
        "dynamic_threshold": (100/255, 197/255, 235/255, 0.7),          # Light blue
        "motor_command": (127/255, 88/255, 175/255, 0.7),               # Purple
        "myelinated": (68/255, 252/255, 215/255, 1),                    # Bright teal 
        "axon": (0.2, 0.2, 0.2, 0.7),                                   # Dark gray for axons
        "not functionally imaged": (0.5, 0.5, 0.5, 0.7),                # Gray for not functionally imaged
    }

    # Line styles for 'Neurotransmitter Classifier'
    NEUROTRANSMITTER_OUTLINE_STYLES = {
        'excitatory': 'solid',   # Default solid outline
        'inhibitory': 'dashed', # Dashed outline for inhibition
        'unknown': 'dotted'     # Dotted outline for unknown
    }

    # Function to adjust luminance for contrast
    def adjust_luminance(rgb, factor=1.5):
        r, g, b = rgb[:3]
        h, l, s = rgb_to_hls(r, g, b)
        l = min(1, max(0, l * factor))  # Ensure luminance stays within bounds
        return hls_to_rgb(h, l, s)

    # Merge probabilities for 'not functionally imaged'
    not_functionally_imaged = data_df[data_df['Functional Classifier'] == 'not functionally imaged']
    if not not_functionally_imaged.empty:
        merged_probability = not_functionally_imaged['Probability'].sum()
        merged_count = not_functionally_imaged['Count'].sum()
        data_df = data_df[data_df['Functional Classifier'] != 'not functionally imaged']  # Remove these rows
        merged_row = {
            'Functional Classifier': 'not functionally imaged',
            'Projection Classifier': 'None',
            'Neurotransmitter Classifier': 'unknown',
            'Probability': merged_probability,
            'Count': merged_count
        }
        data_df = pd.concat([data_df, pd.DataFrame([merged_row])], ignore_index=True)

    # Merge probabilities for 'dynamic_threshold'
    dynamic_threshold = data_df[data_df['Functional Classifier'] == 'dynamic_threshold']
    if not dynamic_threshold.empty:
        merged_probability = dynamic_threshold['Probability'].sum()
        merged_count = dynamic_threshold['Count'].sum()
        data_df = data_df[data_df['Functional Classifier'] != 'dynamic_threshold']  # Remove these rows
        merged_row = {
            'Functional Classifier': 'dynamic_threshold',
            'Projection Classifier': 'None',  # Adjust as appropriate
            'Neurotransmitter Classifier': 'inhibitory',  # Adjust as appropriate
            'Probability': merged_probability,
            'Count': merged_count
        }
        data_df = pd.concat([data_df, pd.DataFrame([merged_row])], ignore_index=True)

    # Number of nodes: 1 in the first layer, rows in the second layer
    layer_sizes = [1, len(data_df)]

    # Compute spacings
    v_spacing = (top - bottom) / float(max(layer_sizes) + 2)
    h_spacing = (right - left) / float(1)  # Only one horizontal space between two layers

    # Initialize node positions
    node_positions = []

    # 1. Plot the input node (Layer 1)
    input_layer_top = bottom + (top - bottom) / 2.0
    input_node_center = (left if connection_type == 'outputs' else right, input_layer_top)
    input_node_radius = node_radius * 4  # Larger size for the single input node

    # Determine the color of the input node
    edge_color = COLOR_CELL_TYPE_DICT.get(input_circle_color, 'gray')

    # Draw the input node
    input_circle = Circle(input_node_center, input_node_radius, edgecolor=edge_color, facecolor=edge_color, lw=3, alpha=0.8)
    ax.add_artist(input_circle)
    node_positions.append([input_node_center])  # Add position for the single input node

    # 2. Plot the output nodes (Layer 2)
    output_layer_positions = []
    output_layer_top = bottom + (top - bottom) / 2.0 + v_spacing * (layer_sizes[1] - 1) / 2.0

    for node_idx, row in data_df.iterrows():
        # Node position
        node_center = (right if connection_type == 'outputs' else left, output_layer_top - node_idx * v_spacing)

        # Determine color based on functional and projection classifiers
        func_class = row['Functional Classifier']
        proj_class = row['Projection Classifier']

        if func_class in COLOR_CELL_TYPE_DICT:
            key = func_class  # Use direct mapping for known functional classifiers
        elif func_class in ['integrator']:
            key = f"{func_class}_{proj_class}"  # Combine with projection classifier for integrator types
        else:
            key = 'not functionally imaged'  # Default fallback for unknown types

        fill_color = COLOR_CELL_TYPE_DICT.get(key, (0.5, 0.5, 0.5, 0.7))  # Default gray if key not found
        outline_color = adjust_luminance(fill_color, factor=0.5)  # Adjust outline luminance
        neurotransmitter_type = row.get('Neurotransmitter Classifier', 'unknown')  # Get neurotransmitter type

        # Node size adjustment based on probabilities
        probability = row['Probability']
        radius = node_radius * (1 + probability * 4)

        # Determine the outline style
        outline_style = NEUROTRANSMITTER_OUTLINE_STYLES.get(neurotransmitter_type, 'solid')

        # Draw the node
        circle = Circle(node_center, radius, edgecolor=outline_color, facecolor=fill_color, 
                        lw=3, alpha=0.8, linestyle=outline_style)  # Thicker outline
        ax.add_artist(circle)

        # Plot the count next to the circle
        count = row['Count'] if 'Count' in row else 0
        label_offset = -radius * 1.5 if connection_type == 'inputs' else radius * 1.5
        ax.text(node_center[0] + label_offset, node_center[1], f"{count}", fontsize=12, ha='left', va='center', color='black', fontname='Arial')

        output_layer_positions.append((node_center, radius))

    node_positions.append(output_layer_positions)

    # 3. Plot connections from the input node to each output node
    for output_idx, (output_pos, output_radius) in enumerate(node_positions[1]):  # Layer 2 nodes
        # Get the connection probability (weights correspond to probabilities)
        probability = data_df.iloc[output_idx]['Probability'] if output_idx < len(data_df) else 0
        size = a * probability + b  # Scale size as ax + b

        # Calculate adjusted connection positions to stop at the edge of circles
        dx = output_pos[0] - input_node_center[0]
        dy = output_pos[1] - input_node_center[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Calculate points on the edges of the input and output circles
        src_edge_x = input_node_center[0] + (input_node_radius / distance) * dx
        src_edge_y = input_node_center[1] + (input_node_radius / distance) * dy
        dst_edge_x = output_pos[0] - (output_radius / distance) * dx
        dst_edge_y = output_pos[1] - (output_radius / distance) * dy

        # Flip directions for inputs
        if connection_type == 'inputs':
            src_edge_x, dst_edge_x = dst_edge_x, src_edge_x
            src_edge_y, dst_edge_y = dst_edge_y, src_edge_y

        # Draw the connection
        line = Line2D([src_edge_x, dst_edge_x], [src_edge_y, dst_edge_y], c='black', lw=size, alpha=0.8)
        ax.add_artist(line)

        # Add arrow, T-bar, or no ending for 'inputs'

        if connection_type == 'inputs':
            
            neurotransmitter_type = data_df.iloc[output_idx]['Neurotransmitter Classifier']
            
            if neurotransmitter_type == 'excitatory':
                arrow_start_x = dst_edge_x - 0.1 * (dst_edge_x - src_edge_x)  # Start slightly before the circle edge
                arrow_start_y = dst_edge_y - 0.1 * (dst_edge_y - src_edge_y)
                ax.arrow(arrow_start_x, arrow_start_y, 
                        (dst_edge_x - arrow_start_x), (dst_edge_y - arrow_start_y),
                        head_width=size * 0.01, head_length=size * 0.01, fc='black', ec='black', length_includes_head=True)
            elif neurotransmitter_type == 'inhibitory':
                # T-bar at the far end, pointing towards the input node
                t_bar_length = size * 0.01
                t_dx = dy / distance  # Perpendicular direction
                t_dy = -dx / distance
                ax.plot([dst_edge_x - t_bar_length * t_dx, dst_edge_x + t_bar_length * t_dx],
                        [dst_edge_y - t_bar_length * t_dy, dst_edge_y + t_bar_length * t_dy], c='black', lw=2)
            # If 'unknown', no ending is added (straight line only)

        else: 
            # Add arrow for excitatory or T-bar for inhibitory
            if input_cell_type == 'excitatory':
                arrow_start_x = dst_edge_x - 0.1 * (dst_edge_x - src_edge_x)  # Start slightly before the circle edge
                arrow_start_y = dst_edge_y - 0.1 * (dst_edge_y - src_edge_y)
                ax.arrow(arrow_start_x, arrow_start_y, 
                        (dst_edge_x - arrow_start_x), (dst_edge_y - arrow_start_y),
                        head_width=size * 0.01, head_length=size * 0.01, fc='black', ec='black', length_includes_head=True)
            else:  # Inhibitory
                t_bar_length = size * 0.01
                t_dx = -dy / distance  # Perpendicular direction
                t_dy = dx / distance
                ax.plot([dst_edge_x - t_bar_length * t_dx, dst_edge_x + t_bar_length * t_dx],
                        [dst_edge_y - t_bar_length * t_dy, dst_edge_y + t_bar_length * t_dy], c='black', lw=2)  

    # Optional: Midline for the diagram
    if show_midline:
        midline_x = (left + right) / 2
        midline_top = top - 0.05  # Keep midline within plot height
        midline_bottom = bottom + 0.05
        ax.plot([midline_x, midline_x], [midline_bottom, midline_top], color='lightgray', linestyle='--', linewidth=1.5)

    # Add legend if required
    if add_legend:
        legend_elements = [
            Line2D([0], [0], color='black', lw=3, linestyle='solid', label='Excitatory'),
            Line2D([0], [0], color='black', lw=3, linestyle='dashed', label='Inhibitory'),
            Line2D([0], [0], color='black', lw=3, linestyle='dotted', label='Unknown')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=False, title="Neurotransmitter Classifier", title_fontsize=12, prop={'family': 'Arial'})

    ax.axis('off')
    ax.set_xlim(left - h_spacing, right + h_spacing)
    ax.set_ylim(bottom - v_spacing * 1.5, top + v_spacing)
    ax.set_aspect('equal', adjustable='datalim')


def find_and_load_axon_meshes(root_dir, search_strings):
    loaded_meshes = []  # List to store loaded meshes
    for root, dirs, files in os.walk(root_dir):
        for folder_name in dirs:
            for search_string in search_strings:
                if search_string in folder_name:
                    folder_path = os.path.join(root, folder_name, "mapped")  # Path to the 'mapped' subfolder
                    obj_file = folder_name + "_axon_mapped.obj"  # File name contains the folder name
                    file_path = os.path.join(folder_path, obj_file)
                    if os.path.exists(file_path):
                        # Load the mesh file using navis
                        mesh = navis.read_mesh(file_path)
                        loaded_meshes.append(mesh)
                        break  # Exit the inner loop if a match is found
    
    return loaded_meshes


def find_and_load_cell_meshes(root_dir, search_strings):
    axon_meshes = []
    dendrite_meshes = []
    soma_meshes = []
    # Walk through the directory tree
    for root, dirs, files in os.walk(root_dir):
        for folder_name in dirs:
            for search_string in search_strings:
                if search_string in folder_name:
                    folder_path = os.path.join(root, folder_name)
                    axon_file = folder_name + "_axon_mapped.obj"
                    dendrite_file = folder_name + "_dendrite_mapped.obj"
                    soma_file = folder_name + "_soma_mapped.obj"
                    
                    axon_path = os.path.join(folder_path, "mapped", axon_file)
                    dendrite_path = os.path.join(folder_path, "mapped", dendrite_file)
                    soma_path = os.path.join(folder_path, "mapped", soma_file)
                     
                    if os.path.exists(axon_path):
                        axon_mesh = navis.read_mesh(axon_path)
                        axon_meshes.append(axon_mesh)
                        axon_meshes[-1].units= 'micrometer'
                    if os.path.exists(dendrite_path):
                        dendrite_mesh = navis.read_mesh(dendrite_path)
                        dendrite_meshes.append(dendrite_mesh)
                        dendrite_meshes[-1].units= 'micrometer'
                    if os.path.exists(soma_path):
                        soma_mesh = navis.read_mesh(soma_path)
                        soma_meshes.append(soma_mesh)
                        soma_meshes[-1].units= 'micrometer'
                    break  # Exit the inner loop if a match is found
    
    return axon_meshes, dendrite_meshes, soma_meshes


def plot_output_connectome(projection, brain_meshes, color_meshes, meshs_out_nc, colors_out_nrec, mesh_all_rec, color_all_rec, which_brs, path_to_data, seed_cell_funct_id):
    if projection == "z":
        view = ('x', "-y")  # Set the 2D view to the X-Y plane for Z projection.
        ylim = [-850, -50]  # Define the Y-axis limits for the Z projection.
    elif projection == 'y':
        view = ('x', "z")  # Set the 2D view to the X-Z plane for Y projection.
        ylim = [-30, 300]  # Define the Y-axis limits for the Y projection.

    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])
    ax = plt.subplot(gs[1, 0])

    # Plot brain outlines 
    navis.plot2d(brain_meshes, color=color_meshes, volume_outlines=True,
                            alpha=0.2, linewidth=0.5, method='2d', view=view, group_neurons=True,
                            rasterize=True, ax=ax)

    # Optionally fill the background of brain regions with gray for better visibility.
    for mesh in brain_meshes:
        temp_convex_hull = np.array(mesh.to_2d(view=view))
        ax.fill(temp_convex_hull[:, 0], temp_convex_hull[:, 1], c='#F7F7F7', zorder=-1, alpha=1, ec=None)

    # Plot non-rec neurons
    navis.plot2d(meshs_out_nc, color=colors_out_nrec, alpha=0.1, linewidth=0.2,
                 method='2d', view=view, group_neurons=True, rasterize=True, ax=ax,)
    ax.set_aspect('equal')

    # Plot rec neurons
    navis.plot2d(mesh_all_rec, alpha=0.1, linewidth=0.3,
                    method='2d', color=color_all_rec, view=view, 
                    group_neurons=True, rasterize=True, ax=ax,
                    scalebar="20 um")
    ax.set_aspect('equal')

    # Include midline indicator
    ax.axvline(250, color=(0.85, 0.85, 0.85, 0.2), linewidth=0.5, linestyle='--', alpha=0.5, zorder=0)

    # Standardize the plot dimensions.
    plt.xlim(0, 500)  
    plt.ylim(ylim[0], ylim[1])  # Set specific limits for the Y-axis based on the projection.
    ax.set_facecolor('white')  # Set the background color of the plot to white for clarity.
    ax.axis('off')

    # Display the plot
    plt.show()

    # Save plot
    projection_string = projection +"_"+ which_brs + "_proj"  # Create a string to denote the type

    # Create directories for saving the output files if they do not already exist.
    name_time = datetime.now()
    output_dir = path_to_data.joinpath("clem_zfish1").joinpath("connectomes", projection_string, "pdf")
    output_dir.mkdir(parents=True, exist_ok=True) 
    filename = f"{projection_string}_out-connectome_sc_{seed_cell_funct_id}_{name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    output_path = output_dir.joinpath(filename)
    fig.savefig(output_path, dpi=1200)
    print(f"Figure saved successfully at: {output_path}")


def plot_input_connectome(projection, brain_meshes, color_meshes, mesh_axons, color_axons, meshs_out_nc, colors_out_nrec, mesh_all_rec, color_all_rec, which_brs, path_to_data, seed_cell_funct_id):
    if projection == "z":
        view = ('x', "-y")  # Set the 2D view to the X-Y plane for Z projection.
        ylim = [-850, -50]  # Define the Y-axis limits for the Z projection.
    elif projection == 'y':
        view = ('x', "z")  # Set the 2D view to the X-Z plane for Y projection.
        ylim = [-30, 300]  # Define the Y-axis limits for the Y projection.

    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])
    ax = plt.subplot(gs[1, 0])

    # Plot brain outlines 
    navis.plot2d(brain_meshes, color=color_meshes, volume_outlines=True,
                            alpha=0.2, linewidth=0.5, method='2d', view=view, group_neurons=True,
                            rasterize=True, ax=ax)

    # Optionally fill the background of brain regions with gray for better visibility.
    for mesh in brain_meshes:
        temp_convex_hull = np.array(mesh.to_2d(view=view))
        ax.fill(temp_convex_hull[:, 0], temp_convex_hull[:, 1], c='#F7F7F7', zorder=-1, alpha=1, ec=None)

    # Plot axons
    navis.plot2d(mesh_axons, color=color_axons, alpha=1, linewidth=0.5,
                 method='2d', view=view, group_neurons=True, rasterize=True, ax=ax,)
    ax.set_aspect('equal')

    # Plot non-rec neurons
    navis.plot2d(meshs_out_nc, color=colors_out_nrec, alpha=0.1, linewidth=0.2,
                 method='2d', view=view, group_neurons=True, rasterize=True, ax=ax,)
    ax.set_aspect('equal')

    # Plot rec neurons
    navis.plot2d(mesh_all_rec, alpha=0.1, linewidth=0.3,
                    method='2d', color=color_all_rec, view=view, 
                    group_neurons=True, rasterize=True, ax=ax,
                    scalebar="20 um")
    ax.set_aspect('equal')

    # Include midline indicator
    ax.axvline(250, color=(0.85, 0.85, 0.85, 0.2), linewidth=0.5, linestyle='--', alpha=0.5, zorder=0)

    # Standardize the plot dimensions.
    plt.xlim(0, 500)  
    plt.ylim(ylim[0], ylim[1])  # Set specific limits for the Y-axis based on the projection.
    ax.set_facecolor('white')  # Set the background color of the plot to white for clarity.
    ax.axis('off')

    # Display the plot
    plt.show()

    # Save plot
    projection_string = projection +"_"+ which_brs + "_proj"  # Create a string to denote the type

    # Create directories for saving the output files if they do not already exist.
    name_time = datetime.now()
    output_dir = path_to_data.joinpath("clem_zfish1").joinpath("connectomes", projection_string, "pdf")
    output_dir.mkdir(parents=True, exist_ok=True) 
    filename = f"{projection_string}_in-connectome_sc_{seed_cell_funct_id}_{name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    output_path = output_dir.joinpath(filename)
    fig.savefig(output_path, dpi=1200)
    print(f"Figure saved successfully at: {output_path}")


def generate_directional_connectivity_matrix(root_folder, valid_ids, valid_funct_ids, df_w_hemisphere):
    """
    Generate a directional connectivity matrix for functionally imaged neurons.

    Inputs:
        - root_folder: Path to the folder with neuron connectivity data.
        - df_w_hemisphere: DataFrame containing neuron metadata with hemisphere data.
    
    Output:
        - A directional connectivity matrix (inputs vs outputs) with functional IDs as labels.
    """

    # Initialize directional matrix with functional IDs as labels
    connectivity_matrix = pd.DataFrame(0, index=valid_funct_ids, columns=valid_funct_ids)

    # Initialize a set to store globally counted non-zero synapse IDs
    stored_nonzero_synapse_ids = set()

    for source_id, source_func_id in zip(valid_ids, valid_funct_ids):
        # Fetch the connectivity data for the source neuron
        results = get_inputs_outputs_by_hemisphere(root_folder, [source_id], df_w_hemisphere)

        # Fetch the output synapse table
        file_name = f"clem_zfish1_cell_{source_id}_ng_res_presynapses.csv"
        cell_name = f"clem_zfish1_cell_{source_id}"
        output_file_path = os.path.join(root_folder, cell_name, file_name)
        
        outputs_data = pd.read_csv(output_file_path, comment='#', sep=' ', header=None,
                                   names=["partner_cell_id", "x", "y", "z", "synapse_id", "size",
                                          "prediction_status", "validation_status", "date"])
        valid_outputs = outputs_data[outputs_data['validation_status'].str.contains('valid', na=False)]

        # Process OUTPUT connections
        for direction in ["same_side", "different_side"]:
            outputs = results["outputs"]["synapses"][direction]
            if not outputs.empty and "nucleus_id" in outputs.columns:
                # Ensure valid_ids is not empty
                if len(valid_ids) > 0:  # Check that valid_ids is not empty
                    try:
                        # Convert both outputs["nucleus_id"] and valid_ids to integers
                        outputs["nucleus_id"] = outputs["nucleus_id"].astype(int)
                        valid_ids = [int(id) for id in valid_ids]

                        # Filter the outputs DataFrame
                        outputs = outputs[outputs["nucleus_id"].isin(valid_ids)]
                    except ValueError as e:
                        print(f"Error converting nucleus_id or valid_ids to integers: {e}")
                else:
                    print("Warning: valid_ids is empty. No filtering applied.")

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
                        target_func_id = output_row["functional_id"]
                        connectivity_matrix.loc[source_func_id, target_func_id] += num_new_synapses  # Update OUTPUTS ONLY

                        # Add new non-zero synapses to the globally stored set
                        stored_nonzero_synapse_ids.update(new_nonzero_synapses)

        # Fetch the input synapse table
        file_name = f"clem_zfish1_cell_{source_id}_ng_res_postsynapses.csv"
        input_file_path = os.path.join(root_folder, cell_name, file_name)
        
        inputs_data = pd.read_csv(input_file_path, comment='#', sep=' ', header=None,
                                  names=["partner_cell_id", "x", "y", "z", "synapse_id", "size",
                                         "prediction_status", "validation_status", "date"])
        valid_inputs = inputs_data[inputs_data['validation_status'].str.contains('valid', na=False)]

        # Process INPUT connections
        for direction in ["same_side", "different_side"]:
            inputs = results["inputs"]["synapses"][direction]
            if not inputs.empty and "nucleus_id" in inputs.columns:
                
                # Ensure valid_ids is not empty
                if len(valid_ids) > 0:  # Check that valid_ids is not empty
                    try:
                        # Convert both outputs["nucleus_id"] and valid_ids to integers
                        inputs["nucleus_id"] = inputs["nucleus_id"].astype(int)
                        valid_ids = [int(id) for id in valid_ids]

                        # Filter the outputs DataFrame
                        inputs = inputs[inputs["nucleus_id"].isin(valid_ids)]
                    except ValueError as e:
                        print(f"Error converting nucleus_id or valid_ids to integers: {e}")
                else:
                    print("Warning: valid_ids is empty. No filtering applied.")

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
                        source_input_func_id = input_row["functional_id"]
                        connectivity_matrix.loc[source_input_func_id, source_func_id] += num_new_synapses  # Update INPUTS ONLY

                        # Add new non-zero synapses to the globally stored set
                        stored_nonzero_synapse_ids.update(new_nonzero_synapses)

    return connectivity_matrix


def plot_connectivity_matrix(matrix, functional_types, output_path, title="Directional Connectivity Matrix"):
    """
    Plot the directional connectivity matrix with a discrete color bar and functional type bars, including group-based separating lines.
    
    Inputs:
        - matrix: The connectivity matrix (DataFrame).
        - functional_types: A dictionary mapping functional IDs to their types (e.g., "motor_command").
        - title: Title of the plot.
    """
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np

    # Filter functional types to match the updated matrix
    functional_types = {k: v for k, v in functional_types.items() if k in matrix.index}

    # Define the discrete colormap and bounds
    cmap = mcolors.ListedColormap(["white", "blue", "green", "yellow", "pink", "red"])
    bounds = [0, 1, 2, 3, 4, 5, 6]  # Ensure consistent spacing for all categories
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Group rows and columns by functional type
    sorted_indices = sorted(matrix.index, key=lambda x: functional_types.get(x, "unknown"))
    sorted_matrix = matrix.loc[sorted_indices, sorted_indices]
    group_boundaries = []
    last_type = None

    for i, idx in enumerate(sorted_indices):
        current_type = functional_types.get(idx, "unknown")
        if current_type != last_type:
            group_boundaries.append(i - 0.5)  # Add a boundary between groups
            last_type = current_type
    group_boundaries.append(len(sorted_indices) - 0.5)  # Add a boundary for the last group

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(sorted_matrix, cmap=cmap, norm=norm)

    # Color bar below the plot
    cbar = plt.colorbar(
        cax, ax=ax, boundaries=bounds, ticks=[0, 1, 2, 3, 4, 5], spacing="uniform",
        orientation="horizontal", pad=0.1  # Adjust pad for space below the plot
    )
    cbar.set_label("No. of Synapses")

    # Add labels and title
    ax.set_xticks(range(len(sorted_matrix.columns)))
    ax.set_yticks(range(len(sorted_matrix.index)))
    ax.set_xticklabels(sorted_matrix.columns, rotation=90, fontsize=8)  # Adjusted font size for better readability
    ax.set_yticklabels(sorted_matrix.index, fontsize=8)
    ax.set_xlabel("Pre-synaptic (Axons)")
    ax.set_ylabel("Post-synaptic (Dendrites)")
    ax.set_title(title)

    # Add functional type bars
    for i, functional_id in enumerate(sorted_matrix.index):
        # Get the functional type and its color
        functional_type = functional_types.get(functional_id, "not functionally imaged")
        color = COLOR_CELL_TYPE_DICT.get(functional_type, (0.8, 0.8, 0.8, 0.7))  # Default to light gray if not found

        # Draw a rectangle bar on the left
        ax.add_patch(patches.Rectangle((-1.5, i - 0.5), 1, 1, color=color, zorder=2))
        # Draw a rectangle bar on the top
        ax.add_patch(patches.Rectangle((i - 0.5, -1.5), 1, 1, color=color, zorder=2))

    # Add gridlines to separate groups
    for boundary in group_boundaries:
        ax.axhline(boundary, color='black', linewidth=1.5, zorder=3)  # Horizontal boundary
        ax.axvline(boundary, color='black', linewidth=1.5, zorder=3)  # Vertical boundary

    # Adjust axis limits to make the bars visible
    ax.set_xlim(-1.5, len(sorted_matrix.columns) - 0.5)
    ax.set_ylim(len(sorted_matrix.index) - 0.5, -1.5)

    # Adjust layout to avoid cutting labels
    plt.tight_layout()
    # Show and save the figure
    output_pdf_path = os.path.join(output_path, f"connectivity_matrix.pdf")
    plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.show()


def get_partners(functional_id, connectivity_matrix):
    """
    Retrieve and print both input and output partners for a given functional ID from the connectivity matrix.
    
    Inputs:
        - functional_id: The functional ID for which to retrieve input and output partners.
        - connectivity_matrix: The directional connectivity matrix with functional IDs as labels.
    
    Returns:
        - A tuple of two dictionaries:
            - Output partners: {functional_id: synapse_count} for outputs > 0.
            - Input partners: {functional_id: synapse_count} for inputs > 0.
    """
    if functional_id not in connectivity_matrix.index:
        raise ValueError(f"Functional ID {functional_id} not found in the connectivity matrix.")

    # Retrieve the row for outputs
    outputs = connectivity_matrix.loc[functional_id]
    output_partners = outputs[outputs > 0]

    # Retrieve the column for inputs
    inputs = connectivity_matrix[functional_id]
    input_partners = inputs[inputs > 0]

    # Print outputs
    if output_partners.empty:
        print(f"No output partners found for functional ID {functional_id}.")
    else:
        print(f"Output partners for functional ID {functional_id}:")
        for partner, count in output_partners.items():
            print(f"  - Partner: {partner}, Synapse Count: {count}")

    # Print inputs
    if input_partners.empty:
        print(f"No input partners found for functional ID {functional_id}.")
    else:
        print(f"Input partners for functional ID {functional_id}:")
        for partner, count in input_partners.items():
            print(f"  - Partner: {partner}, Synapse Count: {count}")

    # Return as dictionaries
    return output_partners.to_dict(), input_partners.to_dict()


def get_outputs_inputs_neurons_old(root_folder, path_all_cells, seed_cell_ids):
    """
    Extract valid input and output neurons for given seed cell IDs.
    """
    all_cells_df = pd.read_csv(path_all_cells)
    all_cells_df['functional classifier'] = all_cells_df['functional classifier'].replace({
        'dynamic threshold': 'dynamic_threshold',
        'motor command': 'motor_command'
    })
    
    output_neurons_list = []
    input_neurons_list = []

    for seed_cell_id in seed_cell_ids:
        pattern_output = f"clem_zfish1_cell_{seed_cell_id}_ng_res_presynapses.csv"
        pattern_inputs = f"clem_zfish1_cell_{seed_cell_id}_ng_res_postsynapses.csv"
        
        outputs_path, inputs_path = None, None

        for root, _, files in os.walk(root_folder):
            for filename in fnmatch.filter(files, pattern_output):
                outputs_path = os.path.join(root, filename)
            for filename in fnmatch.filter(files, pattern_inputs):
                inputs_path = os.path.join(root, filename)
        
        if not outputs_path or not inputs_path:
            print(f"Output or input path not found for seed cell ID: {seed_cell_id}")
            continue

        dendrites_column = all_cells_df.iloc[:, 8]  
        axons_column = all_cells_df.iloc[:, 7]  

        # Extract valid outputs
        df = pd.read_csv(outputs_path, comment='#', sep=' ', header=None,
                         names=["partner_cell_id", "x", "y", "z", "synapse_id", "size", 
                                "prediction_status", "validation_status", "date"])
        valid_outputs = df[df.apply(lambda row: row.astype(str).str.contains('valid').any(), axis=1)]
        outputs_IDs = valid_outputs.iloc[:, 0]
        traced_dendrites = set(dendrites_column).intersection(set(outputs_IDs))
        output_neurons = all_cells_df[all_cells_df.iloc[:, 8].isin(traced_dendrites)].iloc[:, :12]
        output_neurons_list.append(output_neurons)

        # Extract valid inputs
        df = pd.read_csv(inputs_path, comment='#', sep=' ', header=None,
                         names=["partner_cell_id", "x", "y", "z", "synapse_id", "size", 
                                "prediction_status", "validation_status", "date"])
        valid_inputs = df[df.apply(lambda row: row.astype(str).str.contains('valid').any(), axis=1)]
        inputs_IDs = valid_inputs.iloc[:, 0]
        traced_axons = set(axons_column).intersection(set(inputs_IDs))
        input_neurons = all_cells_df[all_cells_df.iloc[:, 7].isin(traced_axons)].iloc[:, :12]
        input_neurons_list.append(input_neurons)

    output_neurons_combined = pd.concat(output_neurons_list).drop_duplicates() if output_neurons_list else pd.DataFrame()
    input_neurons_combined = pd.concat(input_neurons_list).drop_duplicates(subset=['axon_id']) if input_neurons_list else pd.DataFrame()

    return output_neurons_combined, input_neurons_combined


def get_outputs_inputs_neurons(root_folder, path_all_cells, seed_cell_ids, export_folder, seed_id_string=None):
    """
    Extract valid input and output neurons for given seed cell IDs and export their nucleus IDs to two separate text files:
    one for inputs and one for outputs, grouped by 'functional classifier'. For integrator neurons, differentiate based on
    'neurotransmitter classifier' (inhibitory or excitatory) and 'projection classifier' (ipsilateral or contralateral).
    For axons, 'axon_id' is used. The function also counts how many seed cells output and input neurons were pulled from.
    If seed_id_string is provided, it is used for file naming instead of seed_cell_ids.
    """

    # Use the provided string if seed_id_string is given, otherwise convert seed_cell_ids into a string
    seed_ids_str = seed_id_string if seed_id_string else "_".join([str(id) for id in seed_cell_ids])

    # Ensure the export folder exists
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    # Construct the export file paths based on seed_ids_str
    export_output_file_path = os.path.join(export_folder, f"{seed_ids_str}_outputs.txt")
    export_input_file_path = os.path.join(export_folder, f"{seed_ids_str}_inputs.txt")

    all_cells_df = pd.read_csv(path_all_cells)
    all_cells_df['functional classifier'] = all_cells_df['functional classifier'].replace({
        'dynamic threshold': 'dynamic_threshold',
        'motor command': 'motor_command'
    })

    output_neurons_list = []
    input_neurons_list = []

    # Initialize counters for seed cells that successfully pulled output and input neurons
    output_seed_counter = 0
    input_seed_counter = 0

    for seed_cell_id in seed_cell_ids:
        pattern_output = f"clem_zfish1_cell_{seed_cell_id}_ng_res_presynapses.csv"
        pattern_inputs = f"clem_zfish1_cell_{seed_cell_id}_ng_res_postsynapses.csv"
        
        outputs_path, inputs_path = None, None

        # Find the matching presynaptic and postsynaptic files
        for root, _, files in os.walk(root_folder):
            for filename in fnmatch.filter(files, pattern_output):
                outputs_path = os.path.join(root, filename)
            for filename in fnmatch.filter(files, pattern_inputs):
                inputs_path = os.path.join(root, filename)
        
        if not outputs_path or not inputs_path:
            print(f"Output or input path not found for seed cell ID: {seed_cell_id}")
            continue

        dendrites_column = all_cells_df['dendrite_id']  # dendrite_id column
        axons_column = all_cells_df['axon_id']  # axon_id column

        # Extract valid outputs (presynaptic partners)
        df_output = pd.read_csv(outputs_path, comment='#', sep=' ', header=None,
                                names=["partner_cell_id", "x", "y", "z", "synapse_id", "size", 
                                       "prediction_status", "validation_status", "date"])
        valid_outputs = df_output[df_output.apply(lambda row: row.astype(str).str.contains('valid').any(), axis=1)]
        outputs_IDs = valid_outputs['partner_cell_id']
        traced_dendrites = set(dendrites_column).intersection(set(outputs_IDs))
        output_neurons = all_cells_df[all_cells_df['dendrite_id'].isin(traced_dendrites)]

        if not output_neurons.empty:
            output_neurons_list.append(output_neurons)
            output_seed_counter += 1  # Increment output counter if neurons are found

        # Extract valid inputs (postsynaptic partners)
        df_input = pd.read_csv(inputs_path, comment='#', sep=' ', header=None,
                               names=["partner_cell_id", "x", "y", "z", "synapse_id", "size", 
                                      "prediction_status", "validation_status", "date"])
        valid_inputs = df_input[df_input.apply(lambda row: row.astype(str).str.contains('valid').any(), axis=1)]
        inputs_IDs = valid_inputs['partner_cell_id']
        traced_axons = set(axons_column).intersection(set(inputs_IDs))
        input_neurons = all_cells_df[all_cells_df['axon_id'].isin(traced_axons)]

        if not input_neurons.empty:
            input_neurons_list.append(input_neurons)
            input_seed_counter += 1  # Increment input counter if neurons are found

    # Combine output and input neurons
    output_neurons_combined = pd.concat(output_neurons_list).drop_duplicates() if output_neurons_list else pd.DataFrame()
    input_neurons_combined = pd.concat(input_neurons_list).drop_duplicates(subset=['axon_id']) if input_neurons_list else pd.DataFrame()

    # Fill NaN with 'not functionally imaged' to handle missing classifier values
    output_neurons_combined['functional classifier'] = output_neurons_combined['functional classifier'].fillna('not functionally imaged')
    input_neurons_combined['functional classifier'] = input_neurons_combined['functional classifier'].fillna('not functionally imaged')

    # Get unique functional classifiers
    functional_classifiers_outputs = output_neurons_combined['functional classifier'].unique()
    functional_classifiers_inputs = input_neurons_combined['functional classifier'].unique()

    def write_neurons_by_type(file, neurons_combined, classifier):
        """Helper function to write neurons to file, differentiating by 'type' and 'axon_id' for axons."""
        # Write cells (nucleus_id) and axons (axon_id)
        cell_ids = neurons_combined[neurons_combined['type'] == 'cell']['nucleus_id'].tolist()
        axon_ids = neurons_combined[neurons_combined['type'] == 'axon']['axon_id'].tolist()

        if cell_ids:
            file.write(f"{classifier} Cells:\n")
            file.write("\n".join(map(str, cell_ids)) + "\n\n")

        if axon_ids:
            file.write(f"{classifier} Axons:\n")
            file.write("\n".join(map(str, axon_ids)) + "\n\n")

    # Write output neurons to a text file
    with open(export_output_file_path, 'w') as output_file:
        for classifier in functional_classifiers_outputs:
            if classifier == 'integrator':
                # Further differentiate integrator neurons
                integrator_neurons = output_neurons_combined[output_neurons_combined['functional classifier'] == 'integrator']

                # Write integrator inhibitory ipsilateral
                integrator_inhib_ipsi = integrator_neurons[
                    (integrator_neurons['neurotransmitter classifier'] == 'inhibitory') &
                    (integrator_neurons['projection classifier'] == 'ipsilateral')
                ]
                if not integrator_inhib_ipsi.empty:
                    write_neurons_by_type(output_file, integrator_inhib_ipsi, 'Integrator Inhibitory Ipsilateral')

                # Write integrator inhibitory contralateral
                integrator_inhib_contra = integrator_neurons[
                    (integrator_neurons['neurotransmitter classifier'] == 'inhibitory') &
                    (integrator_neurons['projection classifier'] == 'contralateral')
                ]
                if not integrator_inhib_contra.empty:
                    write_neurons_by_type(output_file, integrator_inhib_contra, 'Integrator Inhibitory Contralateral')

                # Write integrator excitatory ipsilateral
                integrator_excit_ipsi = integrator_neurons[
                    (integrator_neurons['neurotransmitter classifier'] == 'excitatory') &
                    (integrator_neurons['projection classifier'] == 'ipsilateral')
                ]
                if not integrator_excit_ipsi.empty:
                    write_neurons_by_type(output_file, integrator_excit_ipsi, 'Integrator Excitatory Ipsilateral')

                # Write integrator excitatory contralateral
                integrator_excit_contra = integrator_neurons[
                    (integrator_neurons['neurotransmitter classifier'] == 'excitatory') &
                    (integrator_neurons['projection classifier'] == 'contralateral')
                ]
                if not integrator_excit_contra.empty:
                    write_neurons_by_type(output_file, integrator_excit_contra, 'Integrator Excitatory Contralateral')
            else:
                write_neurons_by_type(output_file, output_neurons_combined[output_neurons_combined['functional classifier'] == classifier], classifier)

    # Write input neurons to a text file
    with open(export_input_file_path, 'w') as input_file:
        for classifier in functional_classifiers_inputs:
            if classifier == 'integrator':
                integrator_neurons = input_neurons_combined[input_neurons_combined['functional classifier'] == 'integrator']

                # Write integrator inhibitory ipsilateral
                integrator_inhib_ipsi = integrator_neurons[
                    (integrator_neurons['neurotransmitter classifier'] == 'inhibitory') &
                    (integrator_neurons['projection classifier'] == 'ipsilateral')
                ]
                if not integrator_inhib_ipsi.empty:
                    write_neurons_by_type(input_file, integrator_inhib_ipsi, 'Integrator Inhibitory Ipsilateral')

                # Write integrator inhibitory contralateral
                integrator_inhib_contra = integrator_neurons[
                    (integrator_neurons['neurotransmitter classifier'] == 'inhibitory') &
                    (integrator_neurons['projection classifier'] == 'contralateral')
                ]
                if not integrator_inhib_contra.empty:
                    write_neurons_by_type(input_file, integrator_inhib_contra, 'Integrator Inhibitory Contralateral')

                # Write integrator excitatory ipsilateral
                integrator_excit_ipsi = integrator_neurons[
                    (integrator_neurons['neurotransmitter classifier'] == 'excitatory') &
                    (integrator_neurons['projection classifier'] == 'ipsilateral')
                ]
                if not integrator_excit_ipsi.empty:
                    write_neurons_by_type(input_file, integrator_excit_ipsi, 'Integrator Excitatory Ipsilateral')

                # Write integrator excitatory contralateral
                integrator_excit_contra = integrator_neurons[
                    (integrator_neurons['neurotransmitter classifier'] == 'excitatory') &
                    (integrator_neurons['projection classifier'] == 'contralateral')
                ]
                if not integrator_excit_contra.empty:
                    write_neurons_by_type(input_file, integrator_excit_contra, 'Integrator Excitatory Contralateral')
            else:
                write_neurons_by_type(input_file, input_neurons_combined[input_neurons_combined['functional classifier'] == classifier], classifier)

    # Return the combined neurons and counters
    return output_neurons_combined, input_neurons_combined, output_seed_counter, input_seed_counter


def find_and_load_axon_meshes(root_dir, search_strings):
    loaded_meshes = []  # List to store loaded meshes
    for root, dirs, files in os.walk(root_dir):
        for folder_name in dirs:
            for search_string in search_strings:
                if search_string in folder_name:
                    folder_path = os.path.join(root, folder_name, "mapped")  # Path to the 'mapped' subfolder
                    obj_file = folder_name + "_axon_mapped.obj"  # File name contains the folder name
                    file_path = os.path.join(folder_path, obj_file)
                    if os.path.exists(file_path):
                        # Load the mesh file using navis
                        mesh = navis.read_mesh(file_path)
                        loaded_meshes.append(mesh)
                        break  # Exit the inner loop if a match is found
    
    return loaded_meshes


def find_and_load_cell_meshes(root_dir, search_strings):
    axon_meshes = []
    dendrite_meshes = []
    soma_meshes = []
    # Walk through the directory tree
    for root, dirs, files in os.walk(root_dir):
        for folder_name in dirs:
            for search_string in search_strings:
                if search_string in folder_name:
                    folder_path = os.path.join(root, folder_name)
                    axon_file = folder_name + "_axon_mapped.obj"
                    dendrite_file = folder_name + "_dendrite_mapped.obj"
                    soma_file = folder_name + "_soma_mapped.obj"
                    
                    axon_path = os.path.join(folder_path, "mapped", axon_file)
                    dendrite_path = os.path.join(folder_path, "mapped", dendrite_file)
                    soma_path = os.path.join(folder_path, "mapped", soma_file)
                     
                    if os.path.exists(axon_path):
                        axon_mesh = navis.read_mesh(axon_path)
                        axon_meshes.append(axon_mesh)
                        axon_meshes[-1].units= 'micrometer'
                    if os.path.exists(dendrite_path):
                        dendrite_mesh = navis.read_mesh(dendrite_path)
                        dendrite_meshes.append(dendrite_mesh)
                        dendrite_meshes[-1].units= 'micrometer'
                    if os.path.exists(soma_path):
                        soma_mesh = navis.read_mesh(soma_path)
                        soma_meshes.append(soma_mesh)
                        soma_meshes[-1].units= 'micrometer'
                    break  # Exit the inner loop if a match is found
    
    return axon_meshes, dendrite_meshes, soma_meshes


def plot_output_connectome(projection, brain_meshes, color_meshes, meshs_out_nc, colors_out_nrec, mesh_all_rec, color_all_rec, which_brs, path_to_data, seed_cell_funct_id):
    if projection == "z":
        view = ('x', "-y")  # Set the 2D view to the X-Y plane for Z projection.
        ylim = [-850, -50]  # Define the Y-axis limits for the Z projection.
    elif projection == 'y':
        view = ('x', "z")  # Set the 2D view to the X-Z plane for Y projection.
        ylim = [-30, 300]  # Define the Y-axis limits for the Y projection.

    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])
    ax = plt.subplot(gs[1, 0])

    # Plot brain outlines 
    navis.plot2d(brain_meshes, color=color_meshes, volume_outlines=True,
                            alpha=0.2, linewidth=0.5, method='2d', view=view, group_neurons=True,
                            rasterize=True, ax=ax)

    # Optionally fill the background of brain regions with gray for better visibility.
    for mesh in brain_meshes:
        temp_convex_hull = np.array(mesh.to_2d(view=view))
        ax.fill(temp_convex_hull[:, 0], temp_convex_hull[:, 1], c='#F7F7F7', zorder=-1, alpha=1, ec=None)

    # Plot non-rec neurons
    navis.plot2d(meshs_out_nc, color=colors_out_nrec, alpha=0.1, linewidth=0.2,
                 method='2d', view=view, group_neurons=True, rasterize=True, ax=ax,)
    ax.set_aspect('equal')

    # Plot rec neurons
    navis.plot2d(mesh_all_rec, alpha=1, linewidth=0.5,
                    method='2d', color=color_all_rec, view=view, 
                    group_neurons=True, rasterize=True, ax=ax,
                    scalebar="20 um")
    ax.set_aspect('equal')

    # Include midline indicator
    ax.axvline(250, color=(0.85, 0.85, 0.85, 0.2), linewidth=0.5, linestyle='--', alpha=0.5, zorder=0)

    # Standardize the plot dimensions.
    plt.xlim(0, 500)  
    plt.ylim(ylim[0], ylim[1])  # Set specific limits for the Y-axis based on the projection.
    ax.set_facecolor('white')  # Set the background color of the plot to white for clarity.
    ax.axis('off')

    # Display the plot
    plt.show()

    # Save plot
    projection_string = projection +"_"+ which_brs + "_proj"  # Create a string to denote the type

    # Create directories for saving the output files if they do not already exist.
    name_time = datetime.now()
    output_dir = path_to_data.joinpath("clem_zfish1").joinpath("connectomes", projection_string, "pdf")
    output_dir.mkdir(parents=True, exist_ok=True) 
    filename = f"{projection_string}_out-connectome_sc_{seed_cell_funct_id}_{name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    output_path = output_dir.joinpath(filename)
    fig.savefig(output_path, dpi=1200)
    print(f"Figure saved successfully at: {output_path}")


def plot_input_connectome(projection, brain_meshes, color_meshes, mesh_axons, color_axons, meshs_out_nc, colors_out_nrec, mesh_all_rec, color_all_rec, which_brs, path_to_data, seed_cell_funct_id):
    if projection == "z":
        view = ('x', "-y")  # Set the 2D view to the X-Y plane for Z projection.
        ylim = [-850, -50]  # Define the Y-axis limits for the Z projection.
    elif projection == 'y':
        view = ('x', "z")  # Set the 2D view to the X-Z plane for Y projection.
        ylim = [-30, 300]  # Define the Y-axis limits for the Y projection.

    fig = plt.figure(figsize=(8, 8))
    gs = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])
    ax = plt.subplot(gs[1, 0])

    # Plot brain outlines 
    navis.plot2d(brain_meshes, color=color_meshes, volume_outlines=True,
                            alpha=0.2, linewidth=0.5, method='2d', view=view, group_neurons=True,
                            rasterize=True, ax=ax)

    # Optionally fill the background of brain regions with gray for better visibility.
    for mesh in brain_meshes:
        temp_convex_hull = np.array(mesh.to_2d(view=view))
        ax.fill(temp_convex_hull[:, 0], temp_convex_hull[:, 1], c='#F7F7F7', zorder=-1, alpha=1, ec=None)

    # Plot axons
    navis.plot2d(mesh_axons, color=color_axons, alpha=1, linewidth=0.5,
                 method='2d', view=view, group_neurons=True, rasterize=True, ax=ax,)
    ax.set_aspect('equal')

    # Plot non-rec neurons
    navis.plot2d(meshs_out_nc, color=colors_out_nrec, alpha=0.1, linewidth=0.2,
                 method='2d', view=view, group_neurons=True, rasterize=True, ax=ax,)
    ax.set_aspect('equal')

    # Plot rec neurons
    navis.plot2d(mesh_all_rec, alpha=1, linewidth=0.5,
                    method='2d', color=color_all_rec, view=view, 
                    group_neurons=True, rasterize=True, ax=ax,
                    scalebar="20 um")
    ax.set_aspect('equal')

    # Include midline indicator
    ax.axvline(250, color=(0.85, 0.85, 0.85, 0.2), linewidth=0.5, linestyle='--', alpha=0.5, zorder=0)

    # Standardize the plot dimensions.
    plt.xlim(0, 500)  
    plt.ylim(ylim[0], ylim[1])  # Set specific limits for the Y-axis based on the projection.
    ax.set_facecolor('white')  # Set the background color of the plot to white for clarity.
    ax.axis('off')

    # Display the plot
    plt.show()

    # Save plot
    projection_string = projection +"_"+ which_brs + "_proj"  # Create a string to denote the type

    # Create directories for saving the output files if they do not already exist.
    name_time = datetime.now()
    output_dir = path_to_data.joinpath("clem_zfish1").joinpath("connectomes", projection_string, "pdf")
    output_dir.mkdir(parents=True, exist_ok=True) 
    filename = f"{projection_string}_in-connectome_sc_{seed_cell_funct_id}_{name_time.strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
    output_path = output_dir.joinpath(filename)
    fig.savefig(output_path, dpi=1200)
    print(f"Figure saved successfully at: {output_path}")


def plot_neuron_activity(ax, neurons, time_axis, hdf_file, direction, color_dict, seed_cell_funct_ids):

    # Loop through each neuron and plot its activity
    for _, neuron in neurons.iterrows():
        if neuron[1] != 'not functionally imaged':
            cell_id = int(neuron[1])
            neuron_group = hdf_file[f"neuron_{cell_id}"]
            avg_activity = neuron_group[f"average_activity_{direction}"][()]
            smooth_activity = savgol_filter(avg_activity, 20, 3)

            color = None
            if neuron[9] == 'integrator' and neuron[11] == 'ipsilateral':
                color = color_dict.get("integrator_ipsilateral")
            elif neuron[9] == 'integrator' and neuron[11] == 'contralateral':
                color = color_dict.get("integrator_contralateral")
            elif neuron[9] == 'dynamic_threshold':
                color = color_dict.get("dynamic_threshold")
            elif neuron[9] == 'motor_command':
                color = color_dict.get("motor_command")

            if cell_id in seed_cell_funct_ids:
                # Plot the seed cell activity with its designated color
                ax.plot(time_axis, smooth_activity, color=color, alpha=0.7, linestyle='-', linewidth=1.5, zorder=10)
            else:
                # Plot non-seed cell activity
                ax.plot(time_axis, smooth_activity, color=color, alpha=0.7, linestyle='-', linewidth=1.5)


def setup_plot(ax, title, time_axis, smooth_activity, all_cells, COLOR_CELL_TYPE_DICT, seed_cell_ids):
    # Loop through all seed_cell_ids and plot each one's activity in its designated color
    for seed_cell_id in seed_cell_ids:
        color = 'black'  # Default color if no match is found

        # Extract the row where the 5th value matches seed_cell_id
        seed_cell_row = all_cells[all_cells.iloc[:, 5] == int(seed_cell_id)]

        if not seed_cell_row.empty:
            seed_cell_row = seed_cell_row.iloc[0]  # Convert the single-row DataFrame to a Series

            # Determine color based on the function of the seed cell
            if seed_cell_row.iloc[9] == 'integrator' and seed_cell_row.iloc[11] == 'ipsilateral':
                color = COLOR_CELL_TYPE_DICT.get("integrator_ipsilateral")
            elif seed_cell_row.iloc[9] == 'integrator' and seed_cell_row.iloc[11] == 'contralateral':
                color = COLOR_CELL_TYPE_DICT.get("integrator_contralateral")
            elif seed_cell_row.iloc[9] == 'dynamic_threshold':
                color = COLOR_CELL_TYPE_DICT.get("dynamic_threshold")
            elif seed_cell_row.iloc[9] == 'motor_command':
                color = COLOR_CELL_TYPE_DICT.get("motor_command")

        # Plot the seed cell activity in the chosen color (no black dashed line)
        ax.plot(time_axis, smooth_activity, color=color, alpha=0.7, linewidth=1.5, zorder=10)

    # Plot decorations
    ax.axvspan(20, 60, color='gray', alpha=0.1)
    ax.set_xlabel('Time (seconds)', fontsize=16, fontname='Arial')
    ax.set_ylabel('Activity', fontsize=16, fontname='Arial')
    ax.set_title(title, fontsize=18, fontname='Arial', weight='bold')  # Slightly larger and bold title
    ax.set_xlim(0, max(time_axis))  # Ensure the same x-axis limit
    ax.set_ylim(-50, 150.0)  # Set consistent y-axis limits for publication quality
    ax.set_aspect('equal')  # Force the aspect ratio to be square
    ax.tick_params(axis='both', which='major', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Arial')
        label.set_fontsize(14)
