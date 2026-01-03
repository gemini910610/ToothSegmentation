from torch import optim

OPTIMIZERS = {
    'Adam': optim.Adam
}

def get_optimizer(model, config):
    return OPTIMIZERS[config.optimizer.name](model.parameters(), **config.optimizer.parameters)
