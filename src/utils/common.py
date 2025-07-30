import os
import joblib
from box.exceptions import BoxValueError

def save_object(file_path, obj):
    """Saves a Python object to a file using joblib."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise e