import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import numpy as np

class SelfAttention(pl.LightningModule):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels,int(channels/4), batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.save_hyperparameters()
    
    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size*2)
        x = x.permute(0, 2, 1)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        attention_value = attention_value.permute(0, 1, 2)
        return attention_value.view(-1, self.channels, size, size)


class AE(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 128, 2, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256,1),
            nn.LeakyReLU(),
        )
        
        self.attention = SelfAttention(256)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128,1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 16, 2, stride=2, padding=0),
        )
        
        self.criterion = nn.MSELoss()
        self.lr = 3e-4
        
    def test_latent_shape(self,input):
        print(f"Input shape: {input.shape}")
        latent = self.encoder(input)
        print(f"Latent shape: {latent.shape}")
        print(f"Flattened latent shape: {latent.view(latent.shape[0], -1).shape}")
        output = self.decoder(self.attention(latent))
        print(f"Output shape: {output.shape}")
    
    def forward(self,x):
        #x, _ = torch.max(self.aggregator(x), dim=2)
        h = self.encoder(x)
        z = self.attention(h)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch):
        pergament = batch['scroll_segment'].float().unsqueeze(1)
        pergament = nn.BatchNorm3d(1,device='cuda')(pergament)
        
        pred = self(pergament.squeeze())
        loss = self.criterion(pred.squeeze(), pergament.squeeze())
        
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        return optimizer