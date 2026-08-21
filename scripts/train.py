from pathlib import Path

from utils.config import load_config
from utils.console import Table


def train(task, fold):
    config = load_config(Path('configs') / f'{task}.toml', fold)

    match task:
        case 'segmentation':
            from segmentation.trainer import Trainer
        case 'keypoints':
            from keypoints.trainer import Trainer

    Table(
        ['Argument', 'Value'],
        ['task', task],
        ['fold', config.fold]
    ).display()

    trainer = Trainer(config)
    trainer.fit(config.num_epochs)

if __name__ == '__main__':
    from tui.apps import TaskFoldApp, parse_args

    def configure_parser(parser):
        parser.add_argument('task', type=str, choices=['segmentation', 'keypoints'])
        parser.add_argument('fold', type=int)

    args = parse_args(TaskFoldApp, configure_parser)

    task = args['task']
    fold = args['fold']

    train(task, fold)
