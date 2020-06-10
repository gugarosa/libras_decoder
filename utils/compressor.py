import tarfile

import utils.constants as c


def tar_file(file_name):
    """Compress a file using .tar.gz.
    Args:
        file_name (str): Name of the file to be compressed.

    """

    # Opens a .tar.gz file with `file_name`
    with tarfile.open(f'{file_name}.tar.gz', "w:gz") as tar:
        # Adds every file in the folder to the tar file
        tar.add(file_name, arcname='.')


def untar_file(file_name):
    """De-compress a file with .tar.gz.

    Args:
        file_name (str): Name of the file to be de-compressed.

    """

    # Opens a .tar.gz file with `file_name`
    with tarfile.open(f'{file_name}.tar.gz', "r:gz") as tar:
        # Extracts all files
        tar.extractall(path=file_name)
