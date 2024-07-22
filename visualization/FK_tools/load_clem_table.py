import pandas as pd
import os
from pathlib import Path
import numpy as np
import navis

def read_to_pandas_row(file_content):
    data_list = file_content.strip().split('\n')

    # Parse the list to create a dictionary
    data_dict = {}
    for item in data_list:
        key, value = item.split("=", 1)  # Split by the first "=" only
        key = key.strip()
        value = value.strip().strip('"')

        # Special handling for list and nan values
        if value.startswith("[") and value.endswith("]"):
            # Convert string list to actual list
            value = eval(value)
        elif value == "nan":
            value = np.nan
        elif value.isdigit():
            # Convert numeric strings to integers
            value = int(value)

        data_dict[key] = value

    # Create a DataFrame row
    df_row = pd.DataFrame([data_dict])
    return df_row


def load_clem_table(path):
    df = None
    for cell in [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]:
        if path.joinpath(cell).joinpath(cell+'_metadata.txt').exists():
            with open(path.joinpath(cell).joinpath(cell+'_metadata.txt'),'r') as f:

                text = f.read()
            temp_row = read_to_pandas_row(text)


            temp_type = temp_row.loc[0,'type']
            if temp_type != 'cell':

                temp_row.loc[0,'cell_name'] =   temp_type + "_" + str(temp_row.loc[0,f'{temp_type}_id'])

            else:

                temp_row.loc[0,'cell_name'] =   temp_type + "_" + str(temp_row.loc[0,f'{temp_type}_name'])

            if df is None:
                df = temp_row
            else:
                df = pd.concat([df,temp_row])
    df.reset_index(drop=True)
    return df







if __name__ == '__main__':
    my_df = load_clem_table(Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\CLEM_paper_data\clem_zfish1\all_cells'),repa)
    print('Success!')
