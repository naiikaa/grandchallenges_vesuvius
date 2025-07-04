import lightning as pl
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch 

class DDPMlt(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Unet(dim = 64,
            channels= 1,
            dim_mults = (1, 2, 4,8),
            flash_attn = False
            )
        self.diffusion = GaussianDiffusion(
            self.model,
            image_size = 32,
            timesteps = 1000  
            )

    def forward(self, x):
        
        return self.diffusion(x)

    def training_step(self, batch):
        pergament = batch['scroll_segment'].float().unsqueeze(1)
        pergament = torch.nn.BatchNorm3d(1,device='cuda')(pergament)
        
        ink_label = batch['ink_label'].float()
        ink_label = torch.nn.BatchNorm2d(1,device='cuda')(ink_label.unsqueeze(1))
        loss = self(ink_label)
        
        
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=3e-4)
        return optimizer
    