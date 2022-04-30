import math
import json
import subprocess
from os.path import split, join

import torch


def get_run_dir(current_file, run_dir="runs"):
    dir_path = split(current_file)[0]
    if dir_path == "":
        save_dir = run_dir
    else:
        save_dir = join(dir_path, run_dir)
    return save_dir