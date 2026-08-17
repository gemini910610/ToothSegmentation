import tomllib
from pathlib import Path

import torch
from omegaconf import MISSING, OmegaConf


def load_config(config_path, fold=MISSING):
    with open(config_path, 'rb') as file:
        content = tomllib.load(file)

    config = OmegaConf.create(content)

    config.fold = fold
    config.split_file_path = Path('splits') / f'{config.split_filename}.json'
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    OmegaConf.set_struct(config, True)

    return config

if __name__ == '__main__':
    from argparse import ArgumentParser

    from omegaconf.dictconfig import DictConfig

    from utils.console import Table

    def get_items(config, prefix=''):
        for key, value in config.items():
            key = f'{prefix}.{key}' if prefix else key
            if isinstance(value, DictConfig):
                yield from get_items(value, key)
            else:
                yield key, value

    parser = ArgumentParser()
    parser.add_argument('task', type=str, choices=['segmentation', 'keypoints'])
    args = parser.parse_args()

    task = args.task

    config = load_config(Path('configs') / f'{task}.toml', fold=1)

    Table(
        ['Parameter', 'Value'],
        *get_items(config)
    ).display()
