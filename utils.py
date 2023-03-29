import pandas as pd
import numpy as np
import my_globals

# Always save main dataset outside of the code repo.
# In this case, my data_dir is "../project_data" parallel to "."

def get_sub_dataset(size:int=5000, random_seed:int = 0):
    """Generate a subset of the main dataset usinga random seed.
    Saves sub-dataset to a diectory outside of and parallel to main directory. 
    
    :param size: size of subset
    :type size: int, optional. Default to be 5000.
    :param random_seed: random seed
    :type random_seed: int, optional. Default to be 0.
    """
    DATA_PATH = "/".join([my_globals.DATA_DIR, my_globals.MAIN_DATA_NAME])
    DATA_PATH = "../project_data/training.1600000.processed.noemoticon.csv"
    data = pd.read_csv(
        DATA_PATH, 
        # encoding = 'ISO-8859-1',
        encoding = "latin1",
        header = None, 
        names = my_globals.HEADER
    )
    idx = np.random.choice(data.index, size)
    data.loc[idx].to_csv(
        "/".join([my_globals.DATA_DIR, f"twitter_seed{random_seed}.csv"]), 
        index = False,
        encoding = "latin1"
    )
    