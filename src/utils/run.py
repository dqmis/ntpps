import numpy as np
import torch


def set_seed(seed):
    """
    Sets required project seeds.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_deterministic(seed):
    """
    Makes model runs deterministic by
    setting seeds and enabling deterministic settings.
    """
    set_seed(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
