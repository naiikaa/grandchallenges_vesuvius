import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from tqdm import tqdm
from vesuvius import Volume
import numpy as np
import matplotlib.pyplot as plt

class InkLabelDataset(torch.utils.data.Dataset):
    def __init__(self, segment_ids,sample_size=1024, volume_depth = 16 ,upper_bound=0.8, lower_bound=0.2):
        assert isinstance(segment_ids, list), f"segment_ids must be a list is {type(segment_ids)} instead"
        assert len(segment_ids) > 0, f"segment_ids list cannot be empty has length {len(segment_ids)}"

        self.sample_size = sample_size
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.volume_depth = volume_depth
        self.sample_count = 0
        self.segment_ids = segment_ids
        
        print(f"Creating InkLabelDataset with {len(segment_ids)} segments...")
        
        self.data = self.get_data(self.segment_ids)
        print(f"Found {len(self.data)} data points for {len(segment_ids)} provided segments.")
        

        
        self.samples = {}
        self.samples = self.create_samples_from_data()[0]
        
        print(f"Created {len(self.samples)} samples from ink labels.")
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
    
        return self.samples[f'{idx}']
    
    def get_data(self, segment_ids):
        # pull labels in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            data = list(pool.map(self.get_data_from_id, segment_ids))
        
        for item in data:
            if item is None:
                data.remove(item)
        
        return data
        
    def get_data_from_id(self, segment_id):
        try:
            volume = Volume(int(segment_id))
            inklabels = volume.inklabel
            pergament = volume[:self.volume_depth,:,:]
        except Exception as e:
            print(f"Error processing segment {segment_id}: {e}")
            self.segment_ids.remove(segment_id)
            return None
        
        if len(inklabels.shape) > 2:
            # If the inklabels are 3D, we take the first slice
            inklabels = inklabels[:,:,0]
        
        
        return {'inklabel': inklabels, 'scrollsegment': pergament, 'segment_id': segment_id}

    def create_samples_from_data(self):
        with mp.Pool(mp.cpu_count()) as pool:
            samples = pool.map(self.create_sample_from_data, self.data)
        
        return samples
        
            

    
    def create_sample_from_data(self, item):
        samples = {}
        ink_label = item['inklabel']
        pergament = item['scroll_segment']
        h, w = ink_label.shape
        
        for i in range(0, h - self.sample_size + 1, self.sample_size):
            for j in range(0, w - self.sample_size + 1, self.sample_size):
                ink_label_sample = ink_label[i:i + self.sample_size, j:j + self.sample_size]
                if self.check_sample(ink_label_sample):
                    pergament_sample = pergament[:,i:i + self.sample_size, j:j + self.sample_size]
                    samples[f'{self.sample_count}'] = {'ink_label': ink_label_sample, 'scroll_segment': pergament_sample, 'segment_id': item['segment_id']}
                    self.sample_count += 1
                    
        print(f"Created {len(samples)} samples from ink label with shapes {ink_label.shape}, {pergament.shape} for segment {item['segment_id']}")
        
        return samples
    
    def check_sample(self, ink_label):
        non_pixels = (ink_label > 100).sum().item()
        total_pixels = self.sample_size*self.sample_size
        if non_pixels / total_pixels < self.upper_bound and non_pixels / total_pixels > self.lower_bound:
            return True
        return False
    
    
    
    def __test__(self):
        # Test the dataset
        print(f"Dataset length: {len(self)}")
        fig , axes = plt.subplots(2, 5)
        for i in range(5):
            sample = self[i]
            print(f"Sample {i}: ink_label shape {sample['ink_label'].shape}, scroll_segment shape {sample['scroll_segment'].shape}")
            print(f"Sample {i} ink_label max value: {sample['ink_label'].max()}")
            print(f"Sample {i} scroll_segment max value: {sample['scroll_segment'].max()}")
            axes[0, i].imshow(sample['ink_label'].numpy(), cmap='gray')
            axes[1, i].imshow(sample['scroll_segment'].numpy()[0,:,:], cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        plt.tight_layout()
        plt.show()