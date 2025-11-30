from torch import optim

from .lookahead import Lookahead


OPTIM_REGISTRY = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'lookahead': Lookahead,
    'radam': optim.RAdam,
    'sgd': optim.SGD,
    # Add more optimizers here as needed
}