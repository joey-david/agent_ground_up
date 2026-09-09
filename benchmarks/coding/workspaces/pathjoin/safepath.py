import os

def safe_join(base, rel):
    return os.path.normpath(os.path.join(base, rel))
