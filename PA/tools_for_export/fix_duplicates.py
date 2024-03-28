import pandas
from tqdm import tqdm

def fix_duplicates(df):  # modfiy swcs so they dont make trouble when creating obj
    import copy

    work_df = copy.deepcopy(df)

    adoptive_parent = {}
    drop_set = set()
    drop_not_set = set()

    for i_outer, item_outer in tqdm(work_df.iterrows(),total=work_df.shape[0],leave=False):

        for i, item in work_df.iterrows():
            if item['node_id'] - 1 not in drop_not_set:
                if (item['x'] == item_outer['x']) and (item['y'] == item_outer['y']) and (item['z'] == item_outer['z']) and (item['node_id'] != item_outer['node_id']):
                    if item_outer['node_id'] in adoptive_parent.keys():
                        while item_outer['node_id'] in adoptive_parent.keys():
                            item_outer['node_id'] = adoptive_parent[item_outer['node_id']]
                    else:
                        adoptive_parent[int(item["node_id"])] = int(item_outer['node_id'])
                    drop_set.add(int(item["node_id"] - 1))
                    drop_not_set.add(int(item_outer['node_id'] - 1))



    work_df.drop(list(drop_set), axis=0, inplace=True)
    work_df.loc[:, 'parent_id'].replace(to_replace=adoptive_parent, inplace=True)

    for i, item in work_df.iterrows():
        if item['parent_id'] in adoptive_parent.keys():
            work_df.loc[i, 'parent_id'] = adoptive_parent[item['parent_id']]

    return work_df