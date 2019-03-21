import torch
import torch.nn as nn
import numpy as np
from transformer.Models import Transformer

class MultiTransformer(nn.Module):
    """
    Class that holds several transformer modules
    with global attention
    """
    def __init__(self, n_modules):
        super(MultiTransformer, self).__init__()

    def forward(self, x):
        pass

