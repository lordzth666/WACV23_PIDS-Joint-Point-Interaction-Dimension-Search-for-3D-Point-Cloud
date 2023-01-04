import pickle
import json
from typing import (
    Optional,
    Any,
)

def maybe_load_pickle_file(
        file_name: Optional[str] = None):
    """Maybe loading data from a pickle file.
    Args:
        file_name (str): File name of the pickled data.
    Returns:
        An unpickled data object."""
    if file_name is None:
        print("Warning: File name is None when loading from pickle. Return a 'None' object...")
    with open(file_name, 'rb') as fp_any:
        data = pickle.load(fp_any)
    return data
    
def maybe_write_json_file(
        data: Any,
        json_file_name: Optional[str]):
    if json_file_name is None:
        print("Warning: Json file name is None. Do nothing ...")
    else:
        with open(json_file_name, 'w') as fp:
            json.dump(data, fp)

def maybe_load_json_file(
        json_file_name: Optional[str],
    ):
    if json_file_name is None:
        print("Warning: Json file name is None. Do nothing ...")
    else:
        with open(json_file_name, 'r') as fp:
            data = json.load(fp)
        return data
