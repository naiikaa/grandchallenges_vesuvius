import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl

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

class AEA_lightning(pl.LightningModule):
    def __init__(self, lr = 3e-4):
        super().__init__()
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.aggregator = nn.Conv3d(1,32,kernel_size=(3,1,1),padding=(1,0,0))
        
        self.encoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            SelfAttention(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            SelfAttention(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 7),
            nn.LeakyReLU(),
        )

        self.attention = SelfAttention(256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 7),
            SelfAttention(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3,stride=2, padding=1,output_padding=1),
            SelfAttention(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1,output_padding=1),
        )

    def saveModel(self, model, path):
        """Saves state of an model for future loading

        Args:
            model (Neural Network): The model that is saved
            path (String): Location where to save it
        """
        torch.save(model.state_dict(), path)

    def forward(self,x):
        x, _ = torch.max(self.aggregator(x), dim=2)
        h = self.encoder(x)
        z = self.attention(h)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch):
        batch_tiles, batch_labels, _ = batch
        pred = self(batch_tiles)
        target = batch_labels.float()
        loss = self.criterion(pred, target)
        
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr)
        return optimizer