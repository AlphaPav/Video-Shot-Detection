
import shutil
import os

def makeDir(result_path):
    folder = os.path.exists(result_path)
    if not folder:
        os.makedirs(result_path)
    else:
        shutil.rmtree(result_path)
        os.makedirs(result_path)
