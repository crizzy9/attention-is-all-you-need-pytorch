with open('/media/1TB/midi_data/1/1_tokens.txt') as f:
    data = f.read()

x = set()
for d in data.split(' '):
    if len(d) == 0:
        continue
    x.add(d)

vocab = list(x)

input_shape = (1, 1600)

# with open('lala.txt', 'w+') as f:
    
from new_dataset import MultiTrackPianoRollDataset
dt = MultiTrackPianoRollDataset()
print(dt[0])
