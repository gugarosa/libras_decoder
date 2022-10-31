import tarfile


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
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=file_name)
