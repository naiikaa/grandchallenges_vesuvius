import sys 
sys.path.append('..')
from ml.inklabel_dataset import InkLabelDataset
import yaml
import lightning as pl
from models.DDpy_cond import Unet, GaussianDiffusion
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

segment_ids = yaml.safe_load(open('../configs/segment_ids.yaml', 'r'))
segments = [segment_ids['segment_ids']['segments'][0]]

SAMPLE_SIZE = 16

dataset = InkLabelDataset(segment_ids=segment_ids['segment_ids']['segments'],sample_size=SAMPLE_SIZE,upper_bound=.6, lower_bound=0.4)
train, test, val = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
train_Loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True,num_workers=27)
test_Loader = torch.utils.data.DataLoader(test, batch_size=16, shuffle=False,num_workers=27)
validation_Loader = torch.utils.data.DataLoader(val, batch_size=16, shuffle=False,num_workers=27)

model = Unet(img_size=SAMPLE_SIZE,
    dim = 64,
    channels= 1,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = SAMPLE_SIZE,
    timesteps = 1000  
)

class LightningWrapper(pl.LightningModule):
    def __init__(self, diffusion):
        super().__init__()
        self.diffusion = diffusion

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        return self.diffusion(x)

    def training_step(self, batch):
        ink_labels = batch['ink_label'].float().unsqueeze(1).cuda()
        ink_labels = nn.BatchNorm2d(1,device='cuda')(ink_labels)
        pergament = batch['scroll_segment'].float().unsqueeze(1).cuda()
        pergament = nn.BatchNorm3d(1,device='cuda')(pergament)
        loss = self.diffusion(ink_labels, pergament.squeeze())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ink_labels = batch['ink_label'].float().unsqueeze(1).cuda()
        ink_labels = nn.BatchNorm2d(1,device='cuda')(ink_labels)
        pergament = batch['scroll_segment'].float().unsqueeze(1).cuda()
        pergament = nn.BatchNorm3d(1,device='cuda')(pergament)
        loss = self.diffusion(ink_labels, pergament.squeeze())
        self.log('train_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        ink_labels = batch['ink_label'].float().unsqueeze(1).cuda()
        ink_labels = nn.BatchNorm2d(1,device='cuda')(ink_labels)
        pergament = batch['scroll_segment'].float().unsqueeze(1).cuda()
        pergament = nn.BatchNorm3d(1,device='cuda')(pergament)
        loss = self.diffusion(ink_labels, pergament.squeeze())
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=3e-4)
        return optimizer
    
lightning_model = LightningWrapper(diffusion)
trainer = pl.Trainer(max_epochs=500, accelerator="gpu", devices=1, callbacks=[lt.pytorch.callbacks.early_stopping.EarlyStopping(monitor="train_loss", patience=15, mode="min")])

trainer.fit(ddpm, train_Loader, validation_Loader)
torch.save(lightning_model.state_dict(), 'ink_label_diffusion.pth')


def plot_samples(samples):
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
    for i, sample in enumerate(samples):
        axes[i].imshow(sample.permute(1, 2, 0).cpu().numpy())
        axes[i].axis('off')
    
# Generate samples
samples = diffusion.sample(5)
# Plot the samples
plot_samples(samples)
plt.savefig('ink_label_samples.png')

test_sample = next(iter(test_Loader))
idx = 3
#stack the same sample 8 times
input_sample = test_sample['scroll_segment'][idx].repeat(8, 1, 1, 1).float().cuda()

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

plt.savefig('ink_label_test_results.png')
