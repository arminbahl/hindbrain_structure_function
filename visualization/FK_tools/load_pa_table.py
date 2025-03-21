import pandas as pd
from pathlib import Path
import os
import glob
def load_pa_table(path):
    pa_table = pd.read_csv(path)
    pa_table = pa_table[(~pa_table['cell_type_labels'].isna()) &
                        (~pa_table['date_of_tracing'].isna())]

    # clean pa table
    for i, cell in pa_table.iterrows():
        pa_table.at[i, 'cell_type_labels'] = cell['cell_type_labels'].replace("[", "").replace("]", "").replace('"', "").split(',')
    # remove all quote chars in strings
    for column in pa_table.columns:
        if column != 'cell_type_labels':
            pa_table[column] = pa_table[column].apply(lambda x: x.replace('"', '') if type(x) == str else x)

    for i, cell in pa_table.iterrows():
        try:
            pa_table.loc[i, 'metadata_path'] = Path(
                glob.glob(os.path.join(path.parent / cell.cell_name / "*_metadata.txt"))[0])
        except:
            pa_table.loc[i, 'metadata_path'] = Path(
                glob.glob(os.path.join(path.parent / cell.cell_name / "*metadata.txt"))[0])
    return pa_table

if __name__ == '__main__':
    pa_table = load_pa_table(r"C:\Users\ag-bahl\Desktop\photoactivation_cells_table.csv")
    print("Success!")