import errno
import os

def mkdir_p(path):
    '''
    Performs the same functionality as mkdir -p in the bash shell.
    An OSError is raised if the directory does not exist and was
    not able to be created.
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


