"""
Miscellaneous helper objects.
"""
import glob
import os




def get_all_files(path, ext=None):
    """
    Args:
        path (str): Path of the folder.
        ext (str): Extension format.

    Returns:
        (list) Names of all the files in given foled.
    """
    ext = '*' if ext is None else ext
    return glob.glob(path+ext)


def list_devices():
    """
    Returns:
        None; Prints available devices for tensorflow / pytorch.
    """
    try:
        from tensorflow.python.client.device_lib import list_local_devices
        return str(list_local_devices())
    except ImportError:
        from pytorch.cuda import get_device_name, current_device
        return str(get_device_name(current_device()))


class Cls:
    """
    Object of which its representation clears screen. Alternative for when Ctrl+L is not supported by the terminal.
    """
    def __init__(self):
        pass

    def __repr__(self):
        os.system('cls')
        return ''


cls = Cls()
