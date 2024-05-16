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


def load_em_table(path):
    df = None
    for cell in os.listdir(path):
        if path.joinpath(cell).joinpath(cell+'_metadata.txt').exists():
            with open(path.joinpath(cell).joinpath(cell+'_metadata.txt'),'r') as f:

                text = f.read()
            if '\n\n' in text:
                text = text.split('\n\n')[0]
            temp_row = read_to_pandas_row(text)



            if df is None:
                df = temp_row
            else:
                df = pd.concat([df,temp_row])
    df.reset_index(drop=True)

    df = df.rename(columns={"name": "bad_name",'id':'cell_name'})
    return df







if __name__ == '__main__':
    my_df1 = load_em_table(Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\clem_paper_data\em_zfish1\data_cell_89189_postsynaptic_partners\output_data'))
    my_df2 = load_em_table(Path(r'C:\Users\ag-bahl\Desktop\hindbrain_structure_function\nextcloud_folder\clem_paper_data\em_zfish1\data_seed_cells\output_data'))
    print('Success!')
