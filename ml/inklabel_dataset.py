import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from tqdm import tqdm
from vesuvius import Volume


class InkLabelDataset(torch.utils.data.Dataset):
    def __init__(self, segment_ids):
        assert isinstance(segment_ids, list), f"segment_ids must be a list is {type(segment_ids)} instead"
        assert len(segment_ids) > 0, f"segment_ids list cannot be empty has length {len(segment_ids)}"
        
        self.segment_ids = segment_ids
        print(f"Creating InkLabelDataset with {len(segment_ids)} segments...")
        self.ink_labels = self.get_ink_labels(self.segment_ids)
        print(f"Found {len(self.ink_labels)} ink labels for {len(segment_ids)} provided segments.")
        
        print(f"Creating samples from ink labels...")
        self.samples = self.create_samples_from_ink_labels(self.ink_labels)
        print(f"Created {len(self.samples)} samples from ink labels.")
        

    def __len__(self):
        return len(self.ink_labels)

    def __getitem__(self, idx):
    
        return x
    
    def get_ink_labels(self, segment_ids):
        # pull labels in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(pool.map(self.get_inklabels_from_id, segment_ids))
        
        return [result for result in results if result is not None]
        
    def get_inklabels_from_id(segment_id):
        try:
            volume = Volume(int(segment_id))
            inklabels = volume.inklabel
        except Exception as e:
            print(f"Error processing segment {segment_id}: {e}")
            return None
        
        return inklabels

    def create_samples_from_ink_labels(self, ink_labels):
        with mp.Pool(mp.cpu_count()) as pool:
            samples = list(pool.map(self.create_sample_from_ink_label, ink_labels))
        return [sample for sample in samples if sample is not None]
    
    def create_sample_from_ink_label(self, ink_label):
        
