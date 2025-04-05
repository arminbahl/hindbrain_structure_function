
import os
import bpy
import pandas as pd 

SCALE_FACTOR = 0.01  # Convert micrometer (µm) units to millimeters (mm) for visibility

# Color dictionary (Now actively used)
color_cell_type_dict = {
    "integrator_ipsilateral": (155, 79, 6, 1),          # Light blue, blender corrected
    "integrator_contralateral": (255, 10, 73, 1),       # Magenta-pink, blender corrected
    "dynamic_threshold": (2, 61, 235, 1),               # Light blue, blender corrected
    "motor_command": (38, 19, 63, 1),                   # Purple, blender corrected 
    "myelinated": (80, 220, 100, 1),                    # Green 
    "axon": (0, 0, 0, 1),                               # Black
    'non_functionally_imaged': (40, 40, 40, 1)          # Light gray
}

def fetch_filtered_ids(df, 
                       col_1_name, condition_1, 
                       col_2_name=None, condition_2=None,
                       col_3_name=None, condition_3=None, 
                       col_4_name=None, condition_4=None,
                       exclude_col=None, exclude_condition=None):
    """
    Fetch unique values from specific columns based on up to four conditions and allow exclusion based on a condition.
    
    Parameters:
        - df: The DataFrame to filter.
        - col_1_name (str): Name of the first column to filter on.
        - condition_1: Condition for the first column.
        - col_2_name (str, optional): Name of the second column to filter on.
        - condition_2 (optional): Condition for the second column.
        - col_3_name (str, optional): Name of the third column to filter on.
        - condition_3 (optional): Condition for the third column.
        - col_4_name (str, optional): Name of the fourth column to filter on.
        - condition_4 (optional): Condition for the fourth column.
        - exclude_col (str, optional): Name of the column to exclude rows based on.
        - exclude_condition (optional): Condition for the exclusion column.
    
    Returns:
        - nuclei_ids (list of str): Unique values from the 'nucleus_id' column.
        - functional_ids (list of str): Unique values from the 'functional_id' column.
        - axon_ids (list of str): Unique values from the 'axon_id' column.
    """
    # Apply the first condition
    filtered_rows = df.loc[df[col_1_name] == condition_1]
    
    # Apply the second condition if provided
    if col_2_name is not None and condition_2 is not None:
        filtered_rows = filtered_rows.loc[filtered_rows[col_2_name] == condition_2]
    
    # Apply the third condition if provided
    if col_3_name is not None and condition_3 is not None:
        filtered_rows = filtered_rows.loc[filtered_rows[col_3_name] == condition_3]
    
    # Apply the fourth condition if provided
    if col_4_name is not None and condition_4 is not None:
        filtered_rows = filtered_rows.loc[filtered_rows[col_4_name] == condition_4]
    
    # Exclude rows based on the exclude condition
    if exclude_col is not None and exclude_condition is not None:
        filtered_rows = filtered_rows.loc[filtered_rows[exclude_col] != exclude_condition]
    
    # Extract unique values and convert to list of strings
    nuclei_ids = filtered_rows['nucleus_id'].drop_duplicates().astype(str).tolist()
    axon_ids = filtered_rows['axon_id'].drop_duplicates().astype(str).tolist()
    functional_ids = filtered_rows['functional_id'].drop_duplicates().astype(str).tolist()

    return nuclei_ids, functional_ids, axon_ids

# Paths
output_csv = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224_with_hemisphere.csv'
root_dir = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/traced_neurons/all_cells_111224"

# Load data
all_cells_xls_hemisphere = pd.read_csv(output_csv)

# --------------------------- 1st PART: Fetch IDs to plot ---------------------------

# Standardize namings 
all_cells_xls_hemisphere['functional classifier'] = all_cells_xls_hemisphere['functional classifier'].replace({
    'dynamic threshold': 'dynamic_threshold',
    'motor command': 'motor_command'
})

# NFI, Cell, w/out Myl
nfi_ids, nfi_fids, nfi_axids = fetch_filtered_ids(
    all_cells_xls_hemisphere,
    col_1_name='type', condition_1='cell',
    col_2_name='functional_id', condition_2='not functionally imaged',
    exclude_col='functional classifier', exclude_condition='myelinated'
)
nfi_color = [color_cell_type_dict["non_functionally_imaged"]]

# NFI, Axon
ax_ids, ax_fids, ax_ids = fetch_filtered_ids(
    all_cells_xls_hemisphere,
    col_1_name='type', condition_1='axon',
    col_2_name='functional_id', condition_2='not functionally imaged',
)
axons_colors = [color_cell_type_dict["axon"]]

# DTs
dt_nuc_ids, dt_ax_fids, dt_ax_ids = fetch_filtered_ids(
    all_cells_xls_hemisphere,
    col_1_name='type', condition_1='cell',
    col_2_name='functional classifier', condition_2='dynamic_threshold',
)
dt_color = [color_cell_type_dict["dynamic_threshold"]]

# IIs
ii_nuc_ids, ii_ax_fids, ii_ax_ids = fetch_filtered_ids(
    all_cells_xls_hemisphere,
    col_1_name='type', condition_1='cell',
    col_2_name='functional classifier', condition_2='integrator',
    col_3_name='projection classifier', condition_3='ipsilateral',
)
ii_color = [color_cell_type_dict["integrator_ipsilateral"]]

# ICs
ic_nuc_ids, ic_ax_fids, ic_ax_ids = fetch_filtered_ids(
    all_cells_xls_hemisphere,
    col_1_name='type', condition_1='cell',
    col_2_name='functional classifier', condition_2='integrator',
    col_3_name='projection classifier', condition_3='contralateral',
)
ic_color = [color_cell_type_dict["integrator_contralateral"]] 

# MCs
mc_nuc_ids, mc_ax_fids, mc_ax_ids = fetch_filtered_ids(
    all_cells_xls_hemisphere,
    col_1_name='type', condition_1='cell',
    col_2_name='functional classifier', condition_2='motor_command',
)
mc_color = [color_cell_type_dict["motor_command"]]

# Connected Myl
myl_nuc_ids = ['576460752757645460', '576460752668089872', '576460752642235135', '576460752788211249', 
                '576460752788222513', '576460752787866417', '576460752335453630', '576460752800437809',
                '576460752740502940', '576460752673996580', '576460752681526841', '576460752633284264'
                '576460752756175764', '576460752676272406']
myl_color = [color_cell_type_dict["myelinated"]]

# FI, Cell, w/out Myl
nfi_ids, nfi_fids, nfi_axids = fetch_filtered_ids(
    all_cells_xls_hemisphere,
    col_1_name='type', condition_1='cell',
    col_2_name='functional_id', condition_2='not functionally imaged',
    exclude_col='functional classifier', exclude_condition='myelinated'
)
nfi_color = [color_cell_type_dict["non_functionally_imaged"]]

# Connected not modulated
nm_nuc_ids = ['1556', '1252', '5710', '9933']

# Count the number of elements 
len(dt_nuc_ids) + len(ii_nuc_ids) + len(ic_nuc_ids) + len(mc_nuc_ids)
len(nfi_ids) + len(ax_ids) + len(myl_nuc_ids) + len(nm_nuc_ids)

# --------------------------- 2nd PART: Plot with adequate color  ---------------------------

def scale_object(obj, scale_factor=SCALE_FACTOR):
    """Scale an object using a given scale factor."""
    obj.scale = (scale_factor, scale_factor, scale_factor)
    print(f"Scaled {obj.name} by {scale_factor} to match Blender's unit system.")

# Function to create a neuron material with a specific color
def create_neuron_material(material_name, base_color):
    """Create a high-contrast neuron material with a specified color."""
    if material_name in bpy.data.materials:
        return bpy.data.materials[material_name]

    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True  # Enable node-based materials

    # Get material nodes
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Create Principled BSDF shader
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    # Convert color from 0-255 range to 0-1 range
    color_rgba = tuple(c / 255.0 for c in base_color[:3]) + (base_color[3],)

    bsdf.inputs["Base Color"].default_value = color_rgba  # Assign color from dictionary
    bsdf.inputs["Roughness"].default_value = 0.3  # Smooth surface

    # Check for "Specular IOR Level" (for Blender 4.x compatibility)
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.5  # Minimal reflection

    # Create Emission node (Optional for glow)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.location = (-200, 100)
    emission.inputs["Color"].default_value = color_rgba  # Use same color as base
    emission.inputs["Strength"].default_value = 0.5

    # Mix Shader (to blend BSDF & Emission)
    mix_shader = nodes.new(type="ShaderNodeMixShader")
    mix_shader.location = (200, 0)

    # Connect nodes
    links.new(bsdf.outputs["BSDF"], mix_shader.inputs[1])
    links.new(emission.outputs["Emission"], mix_shader.inputs[2])

    # Connect to Material Output
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

    return mat

# Function to apply neuron material with axon-specific coloring
def apply_neuron_material(obj, color, category):
    """Apply a neuron material to an object using the assigned color.
    
    - Other parts use their assigned color from `color_cell_type_dict`.
    """
    material_name = f"NeuronMaterial_{category}"
    base_color = color  # Use assigned color for other parts
    mat = create_neuron_material(material_name, base_color)

    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat  # Replace existing material
    else:
        obj.data.materials.append(mat)  # Add new material

def find_and_load_cell_meshes_with_colors(root_dir, segment_ids_found, assigned_color_key, 
                                          color_dict, default_color=(128, 128, 128, 1.0), 
                                          dendrite_black=True):
    """Load OBJ neuron meshes, assign colors from a dictionary, and apply materials in Blender.

    - `segment_ids_found` is a **list of segment IDs as strings**.
    - `assigned_color_key` is a **string category** (e.g., `'motor_command'`).
    - The actual RGBA color is fetched from `color_dict`.
    - Ensures **axons are always black** unless specified otherwise.
    - Only imports `_mapped.obj` files, skipping `_mapped_mirrored.obj`.

    Parameters:
        - root_dir (str): Path to the root directory containing neuron OBJ folders.
        - segment_ids_found (list of str): List of segment IDs as strings.
        - assigned_color_key (str): Neuron category name (e.g., `'motor_command'`).
        - color_dict (dict): Dictionary mapping neuron categories to RGBA colors.
        - default_color (tuple): Fallback color if `assigned_color_key` is not found.
        - dendrite_black (bool): If True, dendrites are set to black.

    Returns:
        - List of imported Blender objects.
    """
    if not os.path.exists(root_dir):
        print(f"Error: The root directory {root_dir} does not exist.")
        return []

    # Look up color from the dictionary
    assigned_color = color_dict.get(assigned_color_key, default_color)

    imported_objects = []

    for root, dirs, _ in os.walk(root_dir):
        for folder_name in dirs:
            folder_segment_id = folder_name.split('_')[-1]  # Extract segment ID as a string

            # Check if the folder corresponds to a segment we need
            if folder_segment_id in segment_ids_found:
                mapped_folder_path = os.path.join(root, folder_name, "mapped")
                if not os.path.exists(mapped_folder_path):
                    continue  # Skip if the mapped folder does not exist

                for part in ["axon", "dendrite", "soma"]:
                    obj_file = f"{folder_name}_{part}_mapped.obj"
                    obj_path = os.path.join(mapped_folder_path, obj_file)

                    if os.path.exists(obj_path):  # Import only _mapped.obj, skipping _mapped_mirrored.obj
                        bpy.ops.wm.obj_import(filepath=obj_path)
                        print(f"Loaded {part} mesh: {obj_file}")

                        imported_obj = bpy.context.selected_objects[0]
                        imported_objects.append(imported_obj)

                        # Scale object for correct visibility
                        scale_object(imported_obj)

                        # Apply neuron material with retrieved `assigned_color`
                        apply_neuron_material(imported_obj, assigned_color, assigned_color_key)

    return imported_objects

def find_and_load_axon_meshes(root_dir, search_strings, assigned_color_key, color_dict, default_color=(0, 0, 0, 1.0)):
    """Load OBJ axon meshes in Blender and assign colors using the same approach as cell meshes.

    - Searches for folders that contain any of the provided `search_strings`.
    - Only imports `_axon_mapped.obj` files, skipping `_mapped_mirrored.obj`.
    - Assigns colors from `color_dict`, ensuring axons default to black if not specified.

    Parameters:
        - root_dir (str): Path to the root directory containing neuron OBJ folders.
        - search_strings (list of str): List of keywords to match folder names.
        - assigned_color_key (str): Neuron category name (e.g., `'axon'`).
        - color_dict (dict): Dictionary mapping neuron categories to RGBA colors.
        - default_color (tuple): Fallback color if `assigned_color_key` is not found.

    Returns:
        - List of imported Blender objects.
    """
    if not os.path.exists(root_dir):
        print(f"Error: The root directory {root_dir} does not exist.")
        return []

    # Look up color from the dictionary, default to black if not found
    assigned_color = color_dict.get(assigned_color_key, default_color)

    imported_objects = []

    for root, dirs, _ in os.walk(root_dir):
        for folder_name in dirs:
            # Check if any search string is in the folder name
            if any(search_string in folder_name for search_string in search_strings):
                mapped_folder_path = os.path.join(root, folder_name, "mapped")
                if not os.path.exists(mapped_folder_path):
                    continue  # Skip if 'mapped' folder does not exist

                obj_file = f"{folder_name}_axon_mapped.obj"
                obj_path = os.path.join(mapped_folder_path, obj_file)

                if os.path.exists(obj_path):  # Import only `_axon_mapped.obj`
                    bpy.ops.import_scene.obj(filepath=obj_path)
                    print(f"Loaded axon mesh: {obj_file}")

                    imported_obj = bpy.context.selected_objects[0]
                    imported_objects.append(imported_obj)

                    # Scale object for correct visibility
                    scale_object(imported_obj)

                    # Apply neuron material with retrieved `assigned_color`
                    apply_neuron_material(imported_obj, assigned_color, assigned_color_key)

    return imported_objects

# Load and assign neuron materials to meshes
# imported_objects = find_and_load_cell_meshes_with_colors(root_dir, ii_nuc_ids, "integrator_ipsilateral", color_cell_type_dict)
# imported_objects = find_and_load_cell_meshes_with_colors(root_dir, ic_nuc_ids, "integrator_contralateral", color_cell_type_dict)
# imported_objects = find_and_load_cell_meshes_with_colors(root_dir, dt_nuc_ids, "dynamic_threshold", color_cell_type_dict)
imported_objects = find_and_load_cell_meshes_with_colors(root_dir, mc_nuc_ids, "motor_command", color_cell_type_dict)
# imported_objects = find_and_load_cell_meshes_with_colors(root_dir, myl_nuc_ids, "myelinated", color_cell_type_dict)

# imported_objects = find_and_load_cell_meshes_with_colors(root_dir, nfi_ids, "non_functionally_imaged", color_cell_type_dict)
# imported_objects = find_and_load_cell_meshes_with_colors(root_dir, ax_ids, "axon", color_cell_type_dict)