import os
import shutil
import subprocess
import sys

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('exp', type=str)
args = parser.parse_args()

experiment_name = args.exp

log_dir = os.path.join('logs', f'{experiment_name}')

if os.path.exists(log_dir):
    raise FileExistsError(f'log folder "{log_dir}" already exists')

os.mkdir(log_dir)
shutil.copyfile('splits/points.json', f'{log_dir}/points.json')

for fold in range(1, 5):
    try:
        command = [sys.executable, '-m', 'keypoint.train', experiment_name, '--fold', str(fold)]
        subprocess.run(command, check=True)
    except (Exception, KeyboardInterrupt):
        exit(1)
