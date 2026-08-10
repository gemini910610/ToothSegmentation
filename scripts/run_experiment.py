import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from utils.config import load_config

parser = ArgumentParser()
parser.add_argument('task', type=str, choices=['segmentation', 'keypoints'])
args = parser.parse_args()

task = args.task

config_path = Path('configs') / f'{task}.toml'
config = load_config(config_path)

log_dir = Path('logs') / config.experiment

if log_dir.exists():
    raise FileExistsError(f'experiment "{config.experiment}" already exists')

log_dir.mkdir(parents=True)
shutil.copyfile(config.split_file_path, log_dir / config.split_file_path.name)
shutil.copyfile(config_path, log_dir / f'{task}.toml')

for fold in range(1, config.num_folds + 1):
    try:
        command = [sys.executable, '-m', 'scripts.train', task, str(fold)]
        subprocess.run(command, check=True)
    except (Exception, KeyboardInterrupt):
        sys.exit(1)
