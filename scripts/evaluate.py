from pathlib import Path

from utils.config import load_config


def evaluate(task, experiment_name):
    config = load_config(Path('logs') / experiment_name / f'{task}.toml')

    match task:
        case 'segmentation':
            from segmentation.evaluate import Evaluator
        case 'keypoints':
            from keypoints.evaluate import Evaluator

    evaluator = Evaluator(config)
    logs = evaluator.evaluate()

    table = evaluator.format(logs)
    table.display()

if __name__ == '__main__':
    from tui.apps import TaskExperimentApp, parse_args

    def configure_parser(parser):
        parser.add_argument('task', type=str, choices=['segmentation', 'keypoints'])
        parser.add_argument('exp', type=str)

    args = parse_args(TaskExperimentApp, configure_parser)

    task = args['task']
    experiment_name = args['exp']

    evaluate(task, experiment_name)
