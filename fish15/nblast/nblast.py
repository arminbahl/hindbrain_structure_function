"""
NBLAST Matrix Visualization Script
==================================
This script processes an NBLAST matrix and visualizes it as a heatmap.
It includes functionality to sort neurons based on functional types and highlight them with corresponding colors.
The resulting plot is saved as a PDF.
"""

import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# Set plot font to Arial
plt.rcParams["font.family"] = "Arial"

# Constants
COLOR_CELL_TYPE_DICT = {
    "integrator_ipsilateral": (254/255, 179/255, 38/255, 0.7),      # Yellow-orange
    "integrator_contralateral": (232/255, 77/255, 138/255, 0.7),    # Magenta-pink
    "dynamic_threshold": (100/255, 197/255, 235/255, 0.7),          # Light blue
    "motor_command": (127/255, 88/255, 175/255, 0.7),               # Purples
    "not functionally imaged": (0.8, 0.8, 0.8, 0.7)                 # Light gray
}
PATH_NBLAST_MATRIX = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/lda_nblast_predictions/nblast/nblast_matrix_clem.pickle" 
# PATH_NBLAST_MATRIX = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/lda_nblast_predictions/nblast/nblast_matrix_fly.pickle" 
OUTPUT_CSV = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/old_batch/all_cells_171124_with_hemisphere.csv"
OUTPUT_PATH = "/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/lda_nblast_predictions/nblast/"

# Functions
def read_pickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def fetch_filtered_ids(df, col_1_index, condition_1, col_2_index=None, condition_2=None):
    """
    Fetch unique values from two specific columns based on one or two conditions.
    
    Returns:
        - Unique values from column 5 (e.g., nucleus IDs).
        - Unique values from column 1 (e.g., functional IDs).
    """
    filtered_rows = df.loc[df.iloc[:, col_1_index] == condition_1]
    if col_2_index is not None and condition_2 is not None:
        filtered_rows = filtered_rows.loc[filtered_rows.iloc[:, col_2_index] == condition_2]
    nuclei_ids = filtered_rows.iloc[:, 5].drop_duplicates()
    functional_ids = filtered_rows.iloc[:, 1].drop_duplicates()
    return nuclei_ids, functional_ids

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 1: Generate NBLAST matrix

if __name__ == "__main__":
    # Load the matrix and prepare the data
    nblast_matrix = read_pickle(PATH_NBLAST_MATRIX)
    neurons = nblast_matrix['neurons']
    matrix = pd.DataFrame(nblast_matrix['matrix'], index=neurons, columns=neurons)

    # Load and filter the cell data
    all_cells_xls_hemisphere = pd.read_csv(OUTPUT_CSV).drop_duplicates(subset='axon_id')

    # Standardize naming conventions
    all_cells_xls_hemisphere['functional classifier'] = all_cells_xls_hemisphere['functional classifier'].replace({
        'dynamic threshold': 'dynamic_threshold',
        'motor command': 'motor_command'
    })

    # Fetch filtered IDs by functional type
    dt_ids_all_nuc, dt_ids_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere, 9, 'dynamic_threshold')
    ic_ids_all_nuc, ic_ids_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere, 9, 'integrator', 11, 'contralateral')
    ii_ids_all_nuc, ii_ids_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere, 9, 'integrator', 11, 'ipsilateral')
    mc_ids_all_nuc, mc_ids_all_fun = fetch_filtered_ids(all_cells_xls_hemisphere, 9, 'motor_command')

    # Replace indices of the matrix with the functional IDs
    all_ids_fun = np.concatenate([ii_ids_all_fun, ic_ids_all_fun, dt_ids_all_fun, mc_ids_all_fun])
    matrix.index = all_ids_fun
    matrix.columns = all_ids_fun

    # Calculate the number of neurons for each type
    neuron_counts = {
        "integrator_ipsilateral": len(ii_ids_all_fun),
        "integrator_contralateral": len(ic_ids_all_fun),
        "dynamic_threshold": len(dt_ids_all_fun),
        "motor_command": len(mc_ids_all_fun),
    }

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(matrix, cmap="RdBu", vmin=-1, vmax=1)  # Adjusted scale to be between -1 and 1

    # Add a color bar
    cbar = plt.colorbar(cax, ax=ax, orientation="horizontal", pad=0.1)
    cbar.set_label("No. of Synapses")

    # Add labels and title
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("Neurons")
    ax.set_ylabel("Neurons")
    ax.set_title("Nblast Matrix")

    # Add functional type bars
    start_idx = 0
    for functional_type, count in neuron_counts.items():
        end_idx = start_idx + count
        color = COLOR_CELL_TYPE_DICT[functional_type]

        # Add colored rectangles for functional type
        for i in range(start_idx, end_idx):
            ax.add_patch(patches.Rectangle((-1.5, i - 0.5), 1, 1, color=color, zorder=2))
            ax.add_patch(patches.Rectangle((i - 0.5, -1.5), 1, 1, color=color, zorder=2))

        # Add boundary lines between groups
        if start_idx > 0:
            ax.axhline(start_idx - 0.5, color='black', linewidth=1.5, zorder=3)
            ax.axvline(start_idx - 0.5, color='black', linewidth=1.5, zorder=3)

        start_idx = end_idx

    # Adjust axis limits and layout
    ax.set_xlim(-1.5, len(matrix.columns) - 0.5)
    ax.set_ylim(len(matrix.index) - 0.5, -1.5)
    plt.tight_layout()

    # Save and show the figure
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_filename = os.path.basename(PATH_NBLAST_MATRIX).replace(".pickle", ".pdf")
    output_pdf_path = os.path.join(OUTPUT_PATH, output_filename)
    plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Step 2: Generate NBLAST confusion matrix 

# Compute mean scores for each class excluding the diagonal
mean_scores = {}
non_diagonal_mask = ~np.eye(matrix.shape[0], dtype=bool)

start_idx = 0
for functional_type, count in neuron_counts.items():
    end_idx = start_idx + count
    submatrix = matrix.iloc[start_idx:end_idx, start_idx:end_idx]
    non_diagonal_values = submatrix.values[non_diagonal_mask[start_idx:end_idx, start_idx:end_idx]]
    mean_scores[functional_type] = non_diagonal_values.mean()
    start_idx = end_idx

# Print mean scores
print("Mean scores for each functional class (excluding the diagonal):")
for functional_type, mean_score in mean_scores.items():
    print(f"{functional_type}: {mean_score:.3f}")

# Compute mean scores across functional classes
mean_scores_across_classes = {}
functional_types = list(neuron_counts.keys())

for i, class_1 in enumerate(functional_types):
    for j, class_2 in enumerate(functional_types):
        if i == j:
            continue  # Skip same-class comparisons
        start_idx_1, end_idx_1 = sum(list(neuron_counts.values())[:i]), sum(list(neuron_counts.values())[:i + 1])
        start_idx_2, end_idx_2 = sum(list(neuron_counts.values())[:j]), sum(list(neuron_counts.values())[:j + 1])
        submatrix = matrix.iloc[start_idx_1:end_idx_1, start_idx_2:end_idx_2]
        mean_scores_across_classes[f"{class_1} vs {class_2}"] = submatrix.values.mean()

# Print mean scores across classes
print("Mean scores across functional classes:")
for pair, mean_score in mean_scores_across_classes.items():
    print(f"{pair}: {mean_score:.3f}")

# Construct a square matrix to store mean scores
matrix_size = len(functional_types)
mean_scores_matrix = np.full((matrix_size, matrix_size), np.nan)

# Fill diagonal with within-class mean scores
for i, class_ in enumerate(functional_types):
    mean_scores_matrix[i, i] = mean_scores[class_]

# Fill off-diagonal with between-class mean scores
for i, class_1 in enumerate(functional_types):
    for j, class_2 in enumerate(functional_types):
        if i != j:
            pair_key = f"{class_1} vs {class_2}"
            if pair_key in mean_scores_across_classes:
                mean_scores_matrix[i, j] = mean_scores_across_classes[pair_key]

# Convert to DataFrame for structured visualization
mean_scores_matrix_df = pd.DataFrame(mean_scores_matrix, index=functional_types, columns=functional_types)

# Plot the heatmap
fig, ax = plt.subplots(figsize=(8, 8))  # Ensure square aspect ratio
cax = ax.matshow(mean_scores_matrix, cmap="RdBu", vmin=-1, vmax=1)

# Add color bar
cbar = plt.colorbar(cax)
cbar.set_label("Mean Nblast Score")

# Set ticks and labels
ax.set_xticks(range(matrix_size))
ax.set_yticks(range(matrix_size))
ax.set_xticklabels(functional_types, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(functional_types, fontsize=10)
ax.set_xlabel("Functional Classes")
ax.set_ylabel("Functional Classes")
ax.set_title("Mean Nblast Scores Across Functional Classes")

# Display numerical values within the matrix
for i in range(matrix_size):
    for j in range(matrix_size):
        text = f"{mean_scores_matrix[i, j]:.2f}" if not np.isnan(mean_scores_matrix[i, j]) else "N/A"
        ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()

# Save the plot as a PDF
output_filename = os.path.basename(PATH_NBLAST_MATRIX).replace(".pickle", "-confusion.pdf")
output_pdf_path = os.path.join(OUTPUT_PATH, output_filename)
plt.savefig(output_pdf_path, format="pdf", dpi=300, bbox_inches="tight")
plt.show()

# %%