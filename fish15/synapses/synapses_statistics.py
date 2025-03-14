
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from matplotlib import rcParams

# Set global font settings
rcParams['font.family'] = 'Arial'

PATH_INPUT = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/traced_neurons/functionally_imaged'
PATH_OUTPUT = '/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/synapses'
path_all_cells='/Users/jonathanboulanger-weill/Dropbox (Harvard University)/hb_connectome/hindbrain_structure_function/clem_zfish1/xls_spreadsheets/all_cells_111224.csv'
df = pd.read_csv(path_all_cells)
df_clean = df.drop_duplicates(subset=['nucleus_id'])

# Define color dictionary
COLOR_CELL_TYPE_DICT = {
    "Integrator Ipsilateral": (254/255, 179/255, 38/255, 0.7),      # Yellow-orange
    "Integrator Contralateral": (232/255, 77/255, 138/255, 0.7),    # Magenta-pink
    "Dynamic Threshold": (100/255, 197/255, 235/255, 0.7),          # Light blue
    "Motor Command": (127/255, 88/255, 175/255, 0.7)                # Purple
}

# First extract IDs of all neuron types 
int_ipsi_ids = df_clean[(df_clean.iloc[:, 9] == 'integrator') & (df_clean.iloc[:, 11].str.contains('ipsilateral'))].iloc[:,5]
int_contra_ids = df_clean[(df_clean.iloc[:, 9] == 'integrator') & (df_clean.iloc[:, 11].str.contains('contralateral'))].iloc[:,5]
dt_ids = df_clean[(df_clean.iloc[:, 9] == 'dynamic threshold')].iloc[:,5]
mc_ids = df_clean[(df_clean.iloc[:, 9] == 'motor command')].iloc[:,5]

##### SYNAPSES NUMBER #####
# %% 

# Function to count 'valid' occurrences in the 7th column using pandas
def count_valid_occurrences(file_path):
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        return df[df.iloc[:,7] == 'valid'].shape[0]  # Count rows where the 7th column contains 'valid'
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 0

# Define a dictionary to store each category of cell IDs
cell_categories = {
    "Integrator Ipsilateral": int_ipsi_ids,
    "Integrator Contralateral": int_contra_ids,
    "Dynamic Threshold": dt_ids,
    "Motor Command": mc_ids
}
# Define figure size in inches based on provided dimensions in mm
width_mm = 40 / 25.4
height_mm = 40 / 25.4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_mm * 2, height_mm))

# Define dictionaries to store counts for each category for postsynaptic and presynaptic data
category_counts_post = { "Integrator Ipsilateral": [], "Integrator Contralateral": [], "Dynamic Threshold": [], "Motor Command": [] }
category_counts_pre = { "Integrator Ipsilateral": [], "Integrator Contralateral": [], "Dynamic Threshold": [], "Motor Command": [] }

# Process each category separately
for category, cell_ids in cell_categories.items():
    # Loop through each cell ID in the current category
    for cell_id in cell_ids.unique():
        cell_id_str = str(cell_id)
        
        # Loop through folders to find files for the cell ID
        for folder_name in os.listdir(PATH_INPUT):
            folder_path = os.path.join(PATH_INPUT, folder_name)
            if os.path.isdir(folder_path) and cell_id_str in folder_name:
                post_file_path = os.path.join(folder_path, f'clem_zfish1_cell_{cell_id_str}_postsynapses.csv')
                pre_file_path = os.path.join(folder_path, f'clem_zfish1_cell_{cell_id_str}_presynapses.csv')
                
                # Count 'valid' occurrences for postsynaptic and presynaptic separately
                valid_count_post = count_valid_occurrences(post_file_path)
                valid_count_pre = count_valid_occurrences(pre_file_path)
                
                # Append counts to respective dictionaries
                category_counts_post[category].append(valid_count_post)
                category_counts_pre[category].append(valid_count_pre)
                break

# Calculate mean and SEM for each category in postsynaptic and presynaptic counts
means_post = {cat: np.mean(counts) for cat, counts in category_counts_post.items()}
sems_post = {cat: sem(counts) for cat, counts in category_counts_post.items()}

means_pre = {cat: np.mean(counts) for cat, counts in category_counts_pre.items()}
sems_pre = {cat: sem(counts) for cat, counts in category_counts_pre.items()}

# Calculate the total number of synapses for each category (postsynaptic and presynaptic)
totals_post = {cat: sum(counts) for cat, counts in category_counts_post.items()}
totals_pre = {cat: sum(counts) for cat, counts in category_counts_pre.items()}

# Calculate the overall totals across all categories
total_synapses_post = sum(totals_post.values())
total_synapses_pre = sum(totals_pre.values())

# Print the results
print("Total number of synapses across cell types:")
print(f"Postsynaptic: {total_synapses_post}")
print(f"Presynaptic: {total_synapses_pre}")

# Font settings
plt.rcParams.update({'font.family': 'Arial', 'font.size': 6})

# Define categories and x-axis positions
categories = list(category_counts_post.keys())
categories_short= ['II', 'CI', 'DT', 'MC']

# Determine common y-axis limits
max_y = max(
    max([max(counts) for counts in category_counts_post.values()] + [0]),
    max([max(counts) for counts in category_counts_pre.values()] + [0])
)
y_limit = (0, max_y + 5)  # Add a little padding

# Plot for postsynaptic data
x = np.arange(len(categories))
ax1.bar(
    x,
    [means_post[cat] for cat in categories],
    yerr=[sems_post[cat] for cat in categories],
    capsize=3,
    color='none',
    edgecolor=[COLOR_CELL_TYPE_DICT[cat] for cat in categories],
    linewidth=1
)
for i, cat in enumerate(categories):
    y_values = category_counts_post[cat]
    x_values = np.full(len(y_values), x[i]) + np.random.normal(0, 0.05, len(y_values))
    ax1.scatter(x_values, y_values, color=COLOR_CELL_TYPE_DICT[cat], edgecolor='black', s=10, alpha=0.7)
ax1.set_ylim(y_limit)
ax1.set_xticks(x)
ax1.set_xticklabels(categories_short, ha="right")
ax1.set_ylabel("Valid Post Count (Mean + SEM)", fontsize=6)
ax1.tick_params(axis='both', which='major', labelsize=6)
ax1.set_title("Postsynaptic Data", fontsize=6)

# Plot for presynaptic data
ax2.bar(
    x,
    [means_pre[cat] for cat in categories],
    yerr=[sems_pre[cat] for cat in categories],
    capsize=3,
    color='none',
    edgecolor=[COLOR_CELL_TYPE_DICT[cat] for cat in categories],
    linewidth=1
)
for i, cat in enumerate(categories):
    y_values = category_counts_pre[cat]
    x_values = np.full(len(y_values), x[i]) + np.random.normal(0, 0.05, len(y_values))
    ax2.scatter(x_values, y_values, color=COLOR_CELL_TYPE_DICT[cat], edgecolor='black', s=10, alpha=0.7)
ax2.set_ylim(y_limit)
ax2.set_xticks(x)
ax2.set_xticklabels(categories_short, ha="right")
ax2.set_ylabel("Valid Pre Count (Mean + SEM)", fontsize=6)
ax2.tick_params(axis='both', which='major', labelsize=6)
ax2.set_title("Presynaptic Data", fontsize=6)

# Show the final bar plot
file_name = 'synapses_clusters.pdf'
file_path = os.path.join(PATH_OUTPUT, file_name)
plt.tight_layout()
#plt.savefig(file_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Figure saved to {file_path}")
plt.show()

##### SYNAPSES SIZE #####
# %% 

# Function to calculate average synapse size per cell for valid synapses
def get_avg_synapse_size(file_path):
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)
        valid_synapse_sizes = df[df.iloc[:, 7] == 'valid'].iloc[:, 5]  # Sizes for 'valid' synapses in 5th column
        if not valid_synapse_sizes.empty:
            return valid_synapse_sizes.mean()  # Average size for valid synapses in this cell
        return np.nan
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return np.nan

# Define a dictionary to store each category of cell IDs
cell_categories = {
    "Integrator Ipsilateral": int_ipsi_ids,
    "Integrator Contralateral": int_contra_ids,
    "Dynamic Threshold": dt_ids,
    "Motor Command": mc_ids
}

# Define figure size in inches based on provided dimensions in mm
width_mm = 40 / 25.4
height_mm = 40 / 25.4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_mm * 2, height_mm))

# Define dictionaries to store average synapse sizes per cell for each category for postsynaptic and presynaptic data
category_avg_sizes_post = { "Integrator Ipsilateral": [], "Integrator Contralateral": [], "Dynamic Threshold": [], "Motor Command": [] }
category_avg_sizes_pre = { "Integrator Ipsilateral": [], "Integrator Contralateral": [], "Dynamic Threshold": [], "Motor Command": [] }

# Process each category separately
for category, cell_ids in cell_categories.items():
    # Loop through each cell ID in the current category
    for cell_id in cell_ids.unique():
        cell_id_str = str(cell_id)
        
        # Loop through folders to find files for the cell ID
        for folder_name in os.listdir(PATH_INPUT):
            folder_path = os.path.join(PATH_INPUT, folder_name)
            if os.path.isdir(folder_path) and cell_id_str in folder_name:
                post_file_path = os.path.join(folder_path, f'clem_zfish1_cell_{cell_id_str}_postsynapses.csv')
                pre_file_path = os.path.join(folder_path, f'clem_zfish1_cell_{cell_id_str}_presynapses.csv')
                
                # Get average synapse size for postsynaptic and presynaptic valid synapses
                avg_size_post = get_avg_synapse_size(post_file_path)
                avg_size_pre = get_avg_synapse_size(pre_file_path)
                
                # Append average sizes to respective dictionaries (only if they are valid numbers)
                if not np.isnan(avg_size_post):
                    category_avg_sizes_post[category].append(avg_size_post)
                if not np.isnan(avg_size_pre):
                    category_avg_sizes_pre[category].append(avg_size_pre)
                break

# Calculate mean and SEM for each category in postsynaptic and presynaptic synapse sizes
means_post = {cat: np.mean(sizes) for cat, sizes in category_avg_sizes_post.items()}
sems_post = {cat: sem(sizes) for cat, sizes in category_avg_sizes_post.items()}

means_pre = {cat: np.mean(sizes) for cat, sizes in category_avg_sizes_pre.items()}
sems_pre = {cat: sem(sizes) for cat, sizes in category_avg_sizes_pre.items()}

# Define categories and x-axis positions
categories = list(category_avg_sizes_post.keys())
categories_short= ['II', 'CI', 'DT', 'MC']

# Determine common y-axis limits
max_y = max(
    max([max(sizes) for sizes in category_avg_sizes_post.values()] + [0]),
    max([max(sizes) for sizes in category_avg_sizes_pre.values()] + [0])
)
y_limit = (0, max_y + 5)  # Add a little padding

# Plot for postsynaptic synapse sizes
x = np.arange(len(categories))
ax1.bar(
    x,
    [means_post[cat] for cat in categories],
    yerr=[sems_post[cat] for cat in categories],
    capsize=3,
    color='none',
    edgecolor=[COLOR_CELL_TYPE_DICT[cat] for cat in categories],
    linewidth=1
)
for i, cat in enumerate(categories):
    y_values = category_avg_sizes_post[cat]
    x_values = np.full(len(y_values), x[i]) + np.random.normal(0, 0.05, len(y_values))
    ax1.scatter(x_values, y_values, color=COLOR_CELL_TYPE_DICT[cat], edgecolor='black', s=10, alpha=0.7)
ax1.set_ylim(y_limit)
ax1.set_xticks(x)
ax1.set_xticklabels(categories_short, ha="right")
ax1.set_ylabel("Synapse Size (Mean + SEM)", fontsize=6)
ax1.tick_params(axis='both', which='major', labelsize=6)
ax1.set_title("Postsynaptic Synapse Sizes", fontsize=6)

# Plot for presynaptic synapse sizes
ax2.bar(
    x,
    [means_pre[cat] for cat in categories],
    yerr=[sems_pre[cat] for cat in categories],
    capsize=3,
    color='none',
    edgecolor=[COLOR_CELL_TYPE_DICT[cat] for cat in categories],
    linewidth=1
)
for i, cat in enumerate(categories):
    y_values = category_avg_sizes_pre[cat]
    x_values = np.full(len(y_values), x[i]) + np.random.normal(0, 0.05, len(y_values))
    ax2.scatter(x_values, y_values, color=COLOR_CELL_TYPE_DICT[cat], edgecolor='black', s=10, alpha=0.7)
ax2.set_ylim(y_limit)
# ax2.set_xticks(x)
ax2.set_xticklabels(categories_short, ha="right")
ax2.set_ylabel("Synapse Size (Mean + SEM)", fontsize=6)
ax2.tick_params(axis='both', which='major', labelsize=6)
ax2.set_title("Presynaptic Synapse Sizes", fontsize=6)

# Show and save the final plot
file_name = 'synapse_sizes_clusters.pdf'
file_path = os.path.join(PATH_OUTPUT, file_name)
plt.tight_layout()
plt.savefig(file_path, dpi=300, bbox_inches='tight', format='pdf')
print(f"Figure saved to {file_path}")
plt.show()

##### STATISTICAL TESTS #####
# %%

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Sizes 
# Combine data for ANOVA
data_post = [(size, "Postsynaptic", cat) for cat, sizes in category_avg_sizes_post.items() for size in sizes]
data_pre = [(size, "Presynaptic", cat) for cat, sizes in category_avg_sizes_pre.items() for size in sizes]
data = pd.DataFrame(data_post + data_pre, columns=["Synapse Size", "Type", "Category"])

# Separate data for ANOVA
post_data = [sizes for sizes in category_avg_sizes_post.values()]
pre_data = [sizes for sizes in category_avg_sizes_pre.values()]

# Perform ANOVA for postsynaptic data
f_stat_post, p_value_post = f_oneway(*post_data)
print(f"ANOVA result for postsynaptic data: F-statistic = {f_stat_post}, p-value = {p_value_post}")

# Perform ANOVA for presynaptic data
f_stat_pre, p_value_pre = f_oneway(*pre_data)
print(f"ANOVA result for presynaptic data: F-statistic = {f_stat_pre}, p-value = {p_value_pre}")

# If ANOVA is significant, perform Tukey's post-hoc test
if p_value_post < 0.05:
    print("\nTukey HSD Post-hoc Test for Postsynaptic Synapse Sizes:")
    tukey_post = pairwise_tukeyhsd(data[data['Type'] == "Postsynaptic"]["Synapse Size"],
                                   data[data['Type'] == "Postsynaptic"]["Category"],
                                   alpha=0.05)
    print(tukey_post)

if p_value_pre < 0.05:
    print("\nTukey HSD Post-hoc Test for Presynaptic Synapse Sizes:")
    tukey_pre = pairwise_tukeyhsd(data[data['Type'] == "Presynaptic"]["Synapse Size"],
                                  data[data['Type'] == "Presynaptic"]["Category"],
                                  alpha=0.05)
    print(tukey_pre)

# Number
# Combine data for ANOVA
data_post = [(size, "Postsynaptic", cat) for cat, sizes in category_counts_post.items() for size in sizes]
data_pre = [(size, "Presynaptic", cat) for cat, sizes in category_counts_pre.items() for size in sizes]
data = pd.DataFrame(data_post + data_pre, columns=["Synapse Number", "Type", "Category"])

# Separate data for ANOVA
post_data = [sizes for sizes in category_counts_post.values()]
pre_data = [sizes for sizes in category_counts_pre.values()]

# Perform ANOVA for postsynaptic data
f_stat_post, p_value_post = f_oneway(*post_data)
print(f"ANOVA result for postsynaptic data: F-statistic = {f_stat_post}, p-value = {p_value_post}")

# Perform ANOVA for presynaptic data
f_stat_pre, p_value_pre = f_oneway(*pre_data)
print(f"ANOVA result for presynaptic data: F-statistic = {f_stat_pre}, p-value = {p_value_pre}")

# If ANOVA is significant, perform Tukey's post-hoc test
if p_value_post < 0.05:
    print("\nTukey HSD Post-hoc Test for Postsynaptic Synapse Number:")
    tukey_post = pairwise_tukeyhsd(data[data['Type'] == "Postsynaptic"]["Synapse Number"],
                                   data[data['Type'] == "Postsynaptic"]["Category"],
                                   alpha=0.05)
    print(tukey_post)

if p_value_pre < 0.05:
    print("\nTukey HSD Post-hoc Test for Presynaptic Synapse Number:")
    tukey_pre = pairwise_tukeyhsd(data[data['Type'] == "Presynaptic"]["Synapse Number"],
                                  data[data['Type'] == "Presynaptic"]["Category"],
                                  alpha=0.05)
    print(tukey_pre)

