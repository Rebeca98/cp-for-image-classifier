
import argparse
import os


def dir_path(string):
    """
    @author: https://stackoverflow.com/a/51212150
    """
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
