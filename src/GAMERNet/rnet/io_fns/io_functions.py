import os
import shutil

def gen_file_list(path: str) -> list:
    return os.listdir(path)

def mv_file(old_path: str, old_filename: str, new_path: str, new_filename: str, slash: bool) -> None:
    """Moves a file from one path to another.

    Parameters
    ----------
    old_path : str
        Path where the initial file is located.
    old_filename : str
        Initial filename.
    new_path : str
        Path where the file is moved to.
    new_filename : str
        New filename for the file.
    slash : bool
        Boolean for whether deciding if slashes have to be added or not.
        If set to True, will add a slash between the path and filename strings.
    """
    if slash == True:
        shutil.move(old_path + "/" + old_filename, new_path + "/" + new_filename)
    else:
        shutil.move(old_path + old_filename, new_path + new_filename)

def rm_dir(path: str) -> None:
    """Removes a directory.

    Parameters
    ----------
    path : str
        Path to the directory to be removed.
    """
    shutil.rmtree(path)