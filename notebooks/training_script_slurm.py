import sys 
sys.path.append('..')
from ml.inklabel_dataset import InkLabelDataset
import yaml
import lightning as pl
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch
import matplotlib.pyplot as plt
import numpy as np

segment_ids = yaml.safe_load(open('../configs/segment_ids.yaml', 'r'))
print(segment_ids['segment_ids'])

SAMPLE_SIZE = 512

ink_label_dataset = InkLabelDataset(segment_ids['segment_ids']['segments'],SAMPLE_SIZE)
ink_label_dataloader = torch.utils.data.DataLoader(ink_label_dataset, batch_size=64, shuffle=True, num_workers=25)

model = Unet(
    dim = 32,
    channels= 1,
    dim_mults = (1, 2, 4),
    flash_attn = False
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

    def training_step(self, batch, batch_idx):
        x = batch
        pred = self(x)
        loss = torch.nn.functional.mse_loss(pred, x)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=3e-4)
        return optimizer
    
lightning_model = LightningWrapper(diffusion)
trainer = pl.Trainer(max_epochs=1000, accelerator="gpu", devices=1, callbacks=[pl.pytorch.callbacks.early_stopping.EarlyStopping(monitor="train_loss", patience=15, mode="min")])

trainer.fit(lightning_model, ink_label_dataloader)
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
