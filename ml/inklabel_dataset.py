import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from tqdm import tqdm
from vesuvius import Volume
import numpy as np

class InkLabelDataset(torch.utils.data.Dataset):
    def __init__(self, segment_ids,sample_size=1024):
        assert isinstance(segment_ids, list), f"segment_ids must be a list is {type(segment_ids)} instead"
        assert len(segment_ids) > 0, f"segment_ids list cannot be empty has length {len(segment_ids)}"

        self.sample_size = sample_size
        self.segment_ids = segment_ids
        print(f"Creating InkLabelDataset with {len(segment_ids)} segments...")
        self.ink_labels = self.get_ink_labels(self.segment_ids)
        print(f"Found {len(self.ink_labels)} ink labels for {len(segment_ids)} provided segments.")
        
        print(f"Creating samples from ink labels...")
        self.samples = self.create_samples_from_ink_labels(self.ink_labels)
        self.samples = [torch.tensor(sample, dtype=torch.float32) for sample in self.samples]
        self.samples = torch.concat(self.samples, dim=0)
        print(f"Created {len(self.samples.shape)}")
        print(f"Created {len(self.samples)} samples from ink labels.")
        

    def __len__(self):
        return len(self.ink_labels)

    def __getitem__(self, idx):
    
        return self.samples[idx]
    
    def get_ink_labels(self, segment_ids):
        # pull labels in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(pool.map(self.get_inklabels_from_id, segment_ids))
        
        return [result for result in results if result is not None]
        
    def get_inklabels_from_id(self, segment_id):
        try:
            volume = Volume(int(segment_id))
            inklabels = volume.inklabel
        except Exception as e:
            print(f"Error processing segment {segment_id}: {e}")
            return None
        
        if len(inklabels.shape) > 2:
            # If the inklabels are 3D, we take the first slice
            inklabels = inklabels[:,:,0]
        
        return inklabels

    def create_samples_from_ink_labels(self, ink_labels):
        with mp.Pool(mp.cpu_count()) as pool:
            samples = list(pool.map(self.create_sample_from_ink_label, ink_labels))
        return [sample for sample in samples if sample is not None]
    
    def create_sample_from_ink_label(self, ink_label):
        samples = []
        h, w = ink_label.shape
        for i in range(0, h - self.sample_size + 1, self.sample_size):
            for j in range(0, w - self.sample_size + 1, self.sample_size):
                sample = ink_label[i:i + self.sample_size, j:j + self.sample_size]
                if sample.max() > 0 and sample.shape[0] == self.sample_size and sample.shape[1] == self.sample_size:
                    samples.append(sample)
        
        return samples
        
