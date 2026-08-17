from pathlib import Path

from utils.config import load_config
from utils.console import Table


def evaluate(task, experiment_name):
    config = load_config(Path('logs') / experiment_name / f'{task}.toml')

    match task:
        case 'segmentation':
            from segmentation.evaluate import Evaluator
        case 'keypoints':
            from keypoints.evaluate import Evaluator

    Table(
        ['Argument', 'Value'],
        ['task', task],
        ['exp', experiment_name]
    ).display()

    evaluator = Evaluator(config)
    logs = evaluator.evaluate()

    table = evaluator.format(logs)
    table.display()

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('task', type=str, choices=['segmentation', 'keypoints'])
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    task = args.task
    experiment_name = args.exp

    evaluate(task, experiment_name)
