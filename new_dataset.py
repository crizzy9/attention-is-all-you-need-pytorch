import torch
import torch.utils.data
import numpy as np
import pretty_midi
import os

MAX_LEN = 37441
NOTES = 128
PITCHES = 128
TRUE_VOCAB_SIZE = NOTES * PITCHES
PAD_TOKEN = TRUE_VOCAB_SIZE 
START_TOKEN = TRUE_VOCAB_SIZE + 1
END_TOKEN = TRUE_VOCAB_SIZE + 2
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN]
VOCAB_SIZE = TRUE_VOCAB_SIZE + len(SPECIAL_TOKENS)


class MultiTrackPianoRollDataset(torch.utils.data.Dataset):
    """Piano Roll dataset."""

    def __init__(self, 
                 root_dir='/media/1TB/midi_data', 
                 tracks=['Bass', 'Drums', 'Guitar', 'Piano', 'Strings'],
                 sample_rate=30,
                 subset_len=None,
                 transform=None):
        """
        Args:
            root_dir (string): Directory with all the pianorolls.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.tracks = tracks
        self.directories = os.listdir(self.root_dir)
        self.transform = transform
        self.sample_rate = sample_rate
        self.subset_len=None
        self.src_vocab_size, self.tgt_vocab_size = VOCAB_SIZE, VOCAB_SIZE
                    
    def __len__(self):
        return self.subset_len if self.subset_len else len(self.directories)

    def __getitem__(self, idx):
        
        sample_dir = self.directories[idx]
        sample_files = [os.path.join(self.root_dir, sample_dir, f'{sample_dir}_{track}.mid') 
                        for track in self.tracks]
        sample = [pretty_midi.PrettyMIDI(sample_file).get_piano_roll(fs=self.sample_rate).T 
                   for sample_file in sample_files]
        
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
def pad_tracks(tracks, pad_token=PAD_TOKEN, pad_to=None):
    new_tracks = []
    max_size = max([t.shape[0] for t in tracks])
    if pad_to:
        assert pad_to >= max_size
        max_size = pad_to
    _, cols = tracks[0].shape
    
    for t in tracks:
        r, c = t.shape
        assert c == cols
        pad = np.ones((max_size - r, cols)) * pad_token
        new_tracks.append(np.concatenate((t, pad), axis=0))
    return new_tracks

def stack_tracks(tracks):
    return np.concatenate([t[...,None] for t in tracks], axis=2)

def concat_tracks(tracks):
    return np.concatenate(tracks, axis=1)

def track_to_tok(tracks):
    return [np.expand_dims(np.argmax(t, axis=1) * 128 + np.max(t, axis=1), axis=1) for t in tracks]

def output_to_pianoroll(model_out):
    clean_output = list(filter(lambda x : x not in SPECIAL_TOKENS, model_out))
    seq_len = len(clean_output)
    pianoroll = np.zeros((seq_len, NOTES))
    for i, val in enumerate(clean_output):
        pianoroll[i][int(val//NOTES)] = val % PITCHES
    return pianoroll

def test_transform(tracks, instrument=2):
    sample = tracks[instrument].flatten().tolist()
    return sample, [START_TOKEN] + sample + [END_TOKEN]
    

transform = lambda x: test_transform(track_to_tok(x))
#     pad_tracks(track_to_tok(x), pad_to=MAX_LEN))