from src.trainer import Trainer
from src.console import Table

def train(config):
    Table(
        ['Argument', 'Value'],
        ['fold', config.fold]
    ).display()

    trainer = Trainer(config)
    trainer.fit(config.num_epochs)

if __name__ == '__main__':
    import os

    from argparse import ArgumentParser
    from src.config import load_config

    config = load_config('configs/config.toml')

    parser = ArgumentParser()
    parser.add_argument('--fold', type=int, choices=list(range(1, config.num_folds + 1)), required=True)
    args = parser.parse_args()

    config.fold = args.fold
    config.split_file_path = os.path.join('splits', f'{config.split_filename}.json')

    train(config)
