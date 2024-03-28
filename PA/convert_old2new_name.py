import pandas as pd
import os

df = pd.read_csv(os.getcwd() + "\photoactivation_cells_table.csv")


df = df[(~df['celltype_labels'].isna())&
                        (~df['date_of_tracing'].isna())]

for i,cell in df.iterrows():
    df.at[i,'celltype_labels'] = cell['celltype_labels'].replace("[","").replace("]","").replace('"',"").split(',')
#remove all quote chars in strings
for column in df.columns:
    if column != 'celltype_labels':
        df[column] = df[column].apply(lambda x: x.replace('"', '') if type(x) == str else x)

query_name = input("Old name?")
query_name = query_name.replace('"',"")

try:
    print("New name:",df.loc[df['photoactivation_ID']==query_name,"cellname"].values[0])
except:
    print('Doesnt exist or cant be found. Are you sure your input is correct?')