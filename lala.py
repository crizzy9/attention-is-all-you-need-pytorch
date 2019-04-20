import torch
import numpy as np
from piano_dataset import MultiTrackPianoRollDataset, transform
from torch.utils.data import DataLoader
dt = MultiTrackPianoRollDataset(transform=transform)

print(len(dt))
print(dt[1])
print(len(dt[1]))
print(len(dt[1][0]))
print(len(dt[1][1]))
print(len(dt[1][2]))
print(len(dt[1][3]))
print(len(dt[1][4]))

