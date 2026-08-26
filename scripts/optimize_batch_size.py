import gc
import json
import subprocess
import sys
from pathlib import Path

import torch

from tui.apps import TaskMaximizeApp, parse_args
from utils.config import load_config


def configure_parser(parser):
    parser.add_argument('task', type=str, choices=['segmentation', 'keypoints'])
    parser.add_argument('--is-train', action='store_true')
    parser.add_argument('--maximize', action='store_true')

def _optimize_batch_size():
    args = parse_args(TaskMaximizeApp, configure_parser)

    task = args['task']
    is_train = args['is_train']
    maximize = args['maximize']

    config_path = Path('configs') / f'{task}.toml'
    config = load_config(config_path, fold=1)
    config.num_workers = 0

    match task:
        case 'segmentation':
            from segmentation.dataset import get_loader
            from segmentation.trainer import Trainer
        case 'keypoints':
            from keypoints.dataset import get_loader
            from keypoints.trainer import Trainer

    trainer = Trainer(config)
    trainer.train_loader.pin_memory = False

    if is_train:
        trainer.model.train()
    else:
        trainer.model.eval()

    batch_size = config.batch_size
    config.batch_size = 1
    loader, _ = get_loader(config)
    for single_batch in loader:
        single_batch = tuple(
            item.to(config.device, non_blocking=True)
            for item in single_batch
        )
        break
    config.batch_size = batch_size

    device = config.device

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    grad_fn = torch.enable_grad if is_train else torch.inference_mode

    success_batch_size = 0
    while True:
        print(f'batch size {config.batch_size}', end='', flush=True)
        try:
            batch = tuple(
                item.repeat_interleave(config.batch_size, dim=0)
                for item in single_batch
            )

            with grad_fn(), torch.autocast(device):
                _, loss = trainer._forward(batch)

            if is_train:
                trainer.optimizer.zero_grad(set_to_none=True)
                trainer.grad_scaler.scale(loss).backward()

            success_batch_size = config.batch_size

            print(' \033[0;92msuccess\033[0m')

            if not maximize:
                break

            config.batch_size *= 2
        except RuntimeError as exception:
            if 'CUDA out of memory' not in str(exception):
                raise

            print(' \033[0;91mCUDA out of memory\033[0m')
            trainer.model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()

            if maximize and success_batch_size:
                break

            config.batch_size //= 2

            if config.batch_size == 0:
                break

    print(json.dumps({
        'batch_size': success_batch_size
    }))

def get_optimal_batch_size(task, is_train=True, maximize=False):
    try:
        command = [sys.executable, '-m', 'scripts.optimize_batch_size', task]
        if is_train:
            command.append('--is-train')
        if maximize:
            command.append('--maximize')
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        result = result.stdout.strip().split('\n')
        return json.loads(result[-1])['batch_size']
    except subprocess.CalledProcessError as exception:
        print(exception.stderr, file=sys.stderr)
        raise

if __name__ == '__main__':
    _optimize_batch_size()
