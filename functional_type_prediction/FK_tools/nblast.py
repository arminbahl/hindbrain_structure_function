import navis
import pandas as pd
import numpy as np
import copy
def nblast_one_group(df,k = 5, resample_size=0.1):

    my_neuron_list = navis.NeuronList(df['swc'],k=k,resample = resample_size)
    dps = navis.make_dotprops(my_neuron_list,k=k,resample = resample_size,progress =False)
    nbl = navis.nblast(dps, dps, progress=False)
    nbl_array = np.array(nbl)
    nbl.index = df.cell_name
    nbl.columns = df.cell_name
    nbl = nbl.iloc[:int(np.floor(len(nbl)/2)),int(np.floor(len(nbl)/2)):]

    return nbl

def nblast_two_groups(df1,df2,k = 5, resample_size=0.1):

    my_neuron_list1 = navis.NeuronList(df1['swc'],k=k,resample = resample_size)
    my_neuron_list2 = navis.NeuronList(df2['swc'], k=k, resample=resample_size)


    dps1 = navis.make_dotprops(my_neuron_list1,k=k,resample = resample_size,progress =False)
    dps2 = navis.make_dotprops(my_neuron_list2, k=k, resample=resample_size,progress =False)

    nbl = navis.nblast(dps1, dps2, progress=False)
    nbl_array = np.array(nbl)
    nbl.index = df1.cell_name
    nbl.columns = df2.cell_name
    return nbl


def nblast_two_groups_custom_matrix(df1,df2,custom_matrix,k = 5, resample_size=0.1,):

    my_neuron_list1 = navis.NeuronList(df1['swc'],k=k,resample = resample_size)
    my_neuron_list2 = navis.NeuronList(df2['swc'], k=k, resample=resample_size)


    dps1 = navis.make_dotprops(my_neuron_list1,k=k,resample = resample_size,progress =False)
    dps2 = navis.make_dotprops(my_neuron_list2, k=k, resample=resample_size,progress =False)

    nbl = navis.nblast(dps1, dps2, progress=False,smat=custom_matrix)
    nbl_array = np.array(nbl)
    nbl.index = df1.cell_name
    nbl.columns = df2.cell_name
    return nbl


def compute_nblast_within_and_between(df,query_keys_input=[]):
    given_categories = []
    query_keys = copy.deepcopy(query_keys_input)

    reverse_cell_type_categories = {
        'ipsilateral': 'morphology',
        'contralateral': 'morphology',
        'inhibitory': 'neurotransmitter',
        'excitatory': 'neurotransmitter',
        'integrator': 'function',
        'dynamic_threshold': 'function',
        'dynamic threshold': 'function',
        'motor_command': 'function',
        'motor command': 'function'
    }

    for key in query_keys:
        given_categories.append(reverse_cell_type_categories[key])

    query_command = ''

    for key in given_categories:
        query_command += f"(df['{key}'].isin(query_keys))&"

    query_command = query_command[:-1]
    if query_command == '':
        raise ValueError('query_keys you have to enter a query_key for compute_nblast_within_and_between to work.')


    df_query = df.loc[eval(query_command),:]
    df_target = df.loc[~(eval(query_command)),:]

    within_values = np.array(nblast_one_group(df_query)).flatten()
    between_values = np.array(nblast_two_groups(df_query,df_target)).flatten()

    print(f"Mean NBLAST within {query_keys_input}: {np.mean(within_values)}")
    print(f"Mean NBLAST between {query_keys_input}: {np.mean(between_values)}\n")

    return within_values,between_values