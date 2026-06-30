from keypoint.trainer import Trainer

def train(experiment_name):
    trainer = Trainer(experiment_name)
    trainer.fit(200)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    experiment_name = args.exp

    train(experiment_name)
