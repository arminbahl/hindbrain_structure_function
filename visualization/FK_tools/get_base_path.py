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
        if user_paths[current_user] == "/YOUR/PATH/TO/CLEM_paper_data/HERE":
            raise NotSetup(
                f"Path isn't configured for user {current_user}. Please modify path_configuration.txt with your path to the CLEM_paper_data which you should download from the nextcloud. A user profile has been created for you.")
    except:
        pass


    try:
        return Path(user_paths[current_user])

    except:
        new_profile = f"{current_user} ?"

        with open(Path(os.getcwd()).joinpath("FK_tools").joinpath('path_configuration.txt'), 'a') as log_file:
            log_file.write(f"\n{current_user} /YOUR/PATH/TO/CLEM_paper_data/HERE")


        raise NotSetup(f"Path isn't configured for user {current_user}. Please modify path_configuration.txt with your path to the CLEM_paper_data which you should download from the nextcloud. A user profile has been created for you.")
