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

SEED = 17
np.random.seed(SEED)
torch.manual_seed(SEED)
            
parser  = argparse.ArgumentParser(description="Train a conditional DDPM model with ink labels.")
parser.add_argument('--sample_size', type=int, default=8, help='Size of the samples to be used in training.')
parser.add_argument('--volume_depth', type=int, default=32, help='Depth of the volume to be used in training.')
parser.add_argument('--upper_bound', type=float, default=0.95, help='Upper bound for the ink label values.')
parser.add_argument('--lower_bound', type=float, default=0.05 , help='Lower bound for the ink label values.')
args = parser.parse_args()

segment_ids = yaml.safe_load(open('../configs/segment_ids.yaml', 'r'))
segments = [segment_ids['segment_ids']['segments'][0]]


dataset = InkLabelDataset(segment_ids=segments,sample_size=args.sample_size,upper_bound=args.upper_bound, lower_bound=args.lower_bound, volume_depth=args.volume_depth, test=False)
test_val = InkLabelDataset(segment_ids=segments,sample_size=args.sample_size,upper_bound=args.upper_bound, lower_bound=args.lower_bound, volume_depth=args.volume_depth, test=True)

#TODO nicht random splitten oder mit selbem seed
test, val = torch.utils.data.random_split(test_val, [0.7, 0.3])

train_Loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,num_workers=27)
test_Loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True,num_workers=27)
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
#trainer.fit(ddpm, train_Loader, validation_Loader)

# create directory and save the model
# check for ./experiments and create it if it does not exist
if not os.path.exists('./experiments'):
    os.makedirs('./experiments')
    
folder_name = f'./experiments/condDDPM_{args.sample_size}_{args.volume_depth}_{args.upper_bound}_{args.lower_bound}'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

#ddpm.save(os.path.join(folder_name, 'model.ckpt'))
ddpm.load(os.path.join(folder_name, 'model.ckpt'))

test_sample = next(iter(test_Loader))
idx = 3
#stack the same sample 8 times
input_sample = test_sample['scroll_segment'][idx].repeat(8, 1, 1, 1).float().cuda()
test_results = ddpm.__test_after_training__(input_sample)

plt.figure(figsize=(16, 4))
for i in range(8):
    plt.subplot(3, 8, i + 1)
    plt.imshow(test_sample['ink_label'][idx,:,:].cpu().numpy(), cmap='gray',vmin=0,vmax=255)
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
def multi_same_exp():
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

def checked_test_loader():
    num_runs = 5
    all_scores = {}
    for i in range(num_runs):
        for batch_numb,test_sample in enumerate(test_Loader):
            batch_scores = []
            input_sample = test_sample['scroll_segment'].float().cuda()
            samples = ddpm.__test_after_training__(input_sample)
            for j in range(input_sample.shape[0]):
                matching_pixels_score = matching_pixels(samples[j].cpu().numpy(), test_sample['ink_label'].squeeze().cpu().numpy()[j])
                batch_scores.append(matching_pixels_score.tolist())
            all_scores['run_' + str(i) + '_batch_' + str(batch_numb)] = batch_scores   
            # save the results to a yaml file
            results_file = os.path.join(folder_name, 'checked_test_loader_run_' + str(i) + '.yaml')
            with open(results_file, 'a') as f:
                yaml.dump(all_scores, f)
            all_scores = {}

def entire_dataset_loader():
    dataset = InkLabelDataset(segment_ids=segments,sample_size=args.sample_size,upper_bound=1.0, lower_bound=0.0, volume_depth=args.volume_depth, test=True)
    data_Loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False,num_workers=27)
    all_scores = {}
    for batch_numb,test_sample in enumerate(data_Loader):
            batch_scores = []
            input_sample = test_sample['scroll_segment'].float().cuda()
            samples = ddpm.__test_after_training__(input_sample)
            for j in range(input_sample.shape[0]):
                matching_pixels_score = matching_pixels(samples[j].cpu().numpy(), test_sample['ink_label'].squeeze().cpu().numpy()[j])
                batch_scores.append(matching_pixels_score.tolist())
            all_scores['batch_' + str(batch_numb)] = batch_scores   
            # save the results to a yaml file
            results_file = os.path.join(folder_name, 'entire_test_dataset.yaml')
            with open(results_file, 'a') as f:
                yaml.dump(all_scores, f)
            all_scores = {}
        
def post_experiment():
    #checked_test_loader()
    entire_dataset_loader()

# release memory
train_Loader = None
validation_Loader = None
dataset = None
test_val = None

post_experiment()
        
        