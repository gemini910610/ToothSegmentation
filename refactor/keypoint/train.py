from refactor.keypoint.trainer import Trainer

def train(experiment_name, fold):
    trainer = Trainer(experiment_name, fold)
    trainer.fit(200)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    parser.add_argument('--fold', type=int, choices=list(range(1, 5)), required=True)
    args = parser.parse_args()

    experiment_name = args.exp
    fold = args.fold

    train(experiment_name, fold)
