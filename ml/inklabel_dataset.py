import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from tqdm import tqdm
from vesuvius import Volume
import numpy as np
import matplotlib.pyplot as plt

class InkLabelDataset(torch.utils.data.Dataset):
    def __init__(self, segment_ids,sample_size=1024, volume_depth = 16 ,upper_bound=0.95, lower_bound=0.05, test=False):
        assert isinstance(segment_ids, list), f"segment_ids must be a list is {type(segment_ids)} instead"
        assert len(segment_ids) > 0, f"segment_ids list cannot be empty has length {len(segment_ids)}"

        self.sample_size = sample_size
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.volume_depth = volume_depth
        self.sample_count = 0
        self.segment_ids = segment_ids
        self.volumes = {}
        self.inklabels = {}
        self.test = test
        self.overall_slices = 0
        self.overall_train_slices = 0
        self.overall_test_slices = 0

        self.check_sample = self.check_sample
        print(f"Creating InkLabelDataset with {len(segment_ids)} segments...")
        
        self.data = self.get_data(self.segment_ids)

        self.samples = {}
        self.samples = self.create_samples_from_data()        
        print(f"Created {len(self.samples)} samples from ink labels.")
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
    
        return self.samples[f'{idx}']
    
    def get_data(self, segment_ids, mp=False):
        # pull labels in parallel
        if mp:
            with mp.Pool(mp.cpu_count()) as pool:
                data = list(pool.map(self.get_data_from_id, segment_ids))
        else:
            data = self.get_data_from_id(segment_ids[0]) 

        print(f"Finished get_data Pool map. With {data}")
        
    def get_data_from_id(self, segment_id):
        print(f"Processing segment {segment_id}...")
        try:
            self.volumes[segment_id] = Volume(int(segment_id))
            self.inklabels[segment_id] = self.volumes[segment_id].inklabel
            print(f"Ink labels loaded")
        except Exception as e:
            print(f"Error processing segment {segment_id}: {e}")
            self.segment_ids.remove(segment_id)
            return 0
        
        if len(self.inklabels[segment_id].shape) > 2:
            # If the inklabels are 3D, we take the first slice
            self.inklabels[segment_id] = self.inklabels[segment_id][:,:,0]
        
        self.inklabels[segment_id] = self.inklabels[segment_id]/255
        # map under 0.5 to 0 and over to 1
        self.inklabels[segment_id] = (self.inklabels[segment_id] > 0.5).astype(np.float32)        
        
        return 1

    def create_samples_from_data(self,mp=False):
        if mp:
            with mp.Pool(mp.cpu_count()) as pool:
                samples = pool.map(self.create_sample_from_data, self.segment_ids)
        else:
            samples = self.create_sample_from_data(self.segment_ids[0])
        
        return samples
        
            

    
    def create_sample_from_data(self, segment_id):
        try:
            samples = {}
            ink_label = self.inklabels[segment_id]
            volume = self.volumes[segment_id]
            h, w = ink_label.shape
            self.overall_slices = h // self.sample_size * w // self.sample_size
            self.overall_train_slices = (h-(h-12000)) // self.sample_size * w // self.sample_size
            self.overall_test_slices = (h-12000) // self.sample_size * w // self.sample_size

            for i in tqdm(range(self.overall_slices)):
                    
                    row , col = divmod(i, w // self.sample_size)
                    
                    if self.test and row*self.sample_size < 12000:
                        continue
                    
                    if not self.test and row*self.sample_size >= 12000:
                        continue

                    ink_label_sample = ink_label[row*self.sample_size:(row+1)*self.sample_size, col*self.sample_size:(col+1)*self.sample_size]
                    if self.check_sample(ink_label_sample):
                        pergament_sample = volume[:self.volume_depth,row*self.sample_size:(row+1)*self.sample_size, col*self.sample_size:(col+1)*self.sample_size]
                        samples[f'{self.sample_count}'] = {'ink_label': ink_label_sample, 'scroll_segment': pergament_sample, 'segment_id': segment_id}
                        self.sample_count += 1
                    
            print(f"Created {len(samples)} samples from ink label with shapes {ink_label.shape}, {pergament_sample.shape} for segment {segment_id}")
            
            return samples
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error creating samples from segment {segment_id}: {e}")
            return {}
    
    def check_sample(self, ink_label):
        total_pixels = self.sample_size*self.sample_size
        if np.sum(ink_label) / total_pixels < self.upper_bound and np.sum(ink_label) / total_pixels > self.lower_bound:
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