import tempfile
import os


def make_temp_path():
    _, path = tempfile.mkstemp()
    os.remove(path)
    return path
