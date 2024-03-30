import os
from pathlib import Path
import pandas as pd
from hindbrain_structure_function.visualization.FK_tools.load_clem_table import read_to_pandas_row

class NotSetup(Exception):
    pass

def get_base_path():
    current_user = Path(os.path.expanduser('~')).name
    user_paths = dict()

    with open(Path(os.getcwd()).joinpath("FK_tools").joinpath('path_configuration.txt'), 'r') as f:
        for line in f:
            parts = line.strip().rstrip().split(' ', 1)
            if len(parts) == 2:
                user, path = parts
                user_paths[user] = path
    try:
        return Path(user_paths[current_user])

    except:
        raise NotSetup(f"Path isn't configured for user {current_user}")
