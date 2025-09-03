import sys
sys.path.append("..")
import torch
import lightning as lt
import yaml
import matplotlib.pyplot as plt
from ml.inklabel_dataset import InkLabelDataset
from models.condDDPM import Diffusion
import argparse
import os
from ml import matching_pixels, matching_pixels_subset_max
import numpy as np
            
parser  = argparse.ArgumentParser(description="Train a conditional DDPM model with ink labels.")
parser.add_argument('--sample_size', type=int, default=16, help='Size of the samples to be used in training.')
parser.add_argument('--volume_depth', type=int, default=16, help='Depth of the volume to be used in training.')
parser.add_argument('--upper_bound', type=float, default=0.6, help='Upper bound for the ink label values.')
parser.add_argument('--lower_bound', type=float, default=0.4, help='Lower bound for the ink label values.')
args = parser.parse_args()

segment_ids = yaml.safe_load(open('../configs/segment_ids.yaml', 'r'))
segments = [segment_ids['segment_ids']['segments'][0]]


dataset = InkLabelDataset(segment_ids=segments,sample_size=args.sample_size,upper_bound=args.upper_bound, lower_bound=args.lower_bound, volume_depth=args.volume_depth)
train, test, val = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
train_Loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True,num_workers=27)
test_Loader = torch.utils.data.DataLoader(test, batch_size=16, shuffle=False,num_workers=27)
validation_Loader = torch.utils.data.DataLoader(val, batch_size=16, shuffle=False,num_workers=27)


ddpm = Diffusion(
    noise_steps=1000,
    beta_start=1e-4,
    beta_end=0.03,
    img_size=args.sample_size,
    volume_depth=args.volume_depth,
    c_in=1,
    c_out=1,
    encoder=None,
    attention=None
)

trainer = lt.Trainer(max_epochs=500, accelerator="gpu", devices=1, callbacks=[lt.pytorch.callbacks.early_stopping.EarlyStopping(monitor="train_loss", patience=15, mode="min")])
trainer.fit(ddpm, train_Loader, validation_Loader)

# create directory and save the model
# check for ./experiments and create it if it does not exist
if not os.path.exists('./experiments'):
    os.makedirs('./experiments')
    
folder_name = f'./experiments/condDDPM_{args.sample_size}_{args.volume_depth}_{args.upper_bound}_{args.lower_bound}'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

ddpm.save(os.path.join(folder_name, 'model.ckpt'))

test_sample = next(iter(test_Loader))
idx = 3
#stack the same sample 8 times
input_sample = test_sample['scroll_segment'][idx].repeat(8, 1, 1, 1).float().cuda()
test_results = ddpm.__test_after_training__(input_sample)

plt.figure(figsize=(16, 4))
for i in range(8):
    plt.subplot(3, 8, i + 1)
    plt.imshow(test_sample['ink_label'][idx].cpu().numpy(), cmap='gray',vmin=0,vmax=255)
    plt.axis('off')
    plt.title("true label")

    plt.subplot(3, 8, i + 9)
    plt.imshow(test_results[i][0].cpu().numpy(), cmap='gray',vmin=0,vmax=255)
    plt.axis('off')
    plt.title("pred by ddpm")
    
    plt.subplot(3,8,i+17)
    plt.imshow(test_sample['ink_label'][idx].cpu().numpy()-test_results[i][0].cpu().numpy(),cmap='gray',vmin=0,vmax=255)
    plt.axis('off')
    plt.title('difference')

plt.savefig(os.path.join(folder_name, 'realToSynth8.png'))

# generate [5, 20, 100, 500, 2000, 10000] samples and check for the pixel score
num_runs = 5
num_samples = [5, 20, 100, 500, 2000, 5000]
results = {}
for i in range(num_runs):
    for n in num_samples:
        input_sample = test_sample['scroll_segment'][0].repeat(n, 1, 1, 1).float().cuda()
        samples = ddpm.__test_after_training__(input_sample)
        
        matching_pixels_score = matching_pixels_subset_max(samples.cpu().numpy(), test_sample['ink_label'].squeeze().cpu().numpy()[0])
        results['run_' + str(i) + '_samples_' + str(n)] = matching_pixels_score.tolist()
    
# save the results to a yaml file
results_file = os.path.join(folder_name, 'results.yaml')
with open(results_file, 'w') as f:
    yaml.dump(results, f)


        