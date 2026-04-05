from keypoint.trainer import Trainer

def train():
    trainer = Trainer()
    trainer.fit(200)

if __name__ == '__main__':
    train()
