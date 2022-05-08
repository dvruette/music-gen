import glob
import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class AudioDataset(Dataset):
    def __init__(self, root_dir_or_files, sr=22050, max_duration=60):
        if isinstance(root_dir_or_files, list):
            self.files = root_dir_or_files
            self.root_dir = None
        else:
            self.root_dir = root_dir_or_files
            self.files = glob.glob(os.path.join(self.root_dir, "**", "*.mp3"), recursive=True)
        
        self.sr = sr
        self.max_duration = max_duration

    def __getitem__(self, i):
        file = self.files[i]
        audio, _ = librosa.load(file, mono=True, sr=self.sr, duration=self.max_duration)
        return torch.from_numpy(audio)

    def __len__(self):
        return len(self.files)

class AudioCollator:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def __call__(self, audio_batch):
        xs = []
        for audio in audio_batch:
            # normalize
            audio = audio / audio.abs().max()
            # extract random chunk of length chunk_size
            if audio.size(0) < self.chunk_size:
                x = F.pad(audio, (0, self.chunk_size - audio.size(0)), value=0.0)
            else:
                start = np.random.randint(audio.size(0) - self.chunk_size)
                x = audio[start:start+self.chunk_size]
            xs.append(x)
        return torch.stack(xs)
