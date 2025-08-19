import os
import torch
from torch import optim
import torch.nn as nn
from models.condUNet import UNet_conditional
import lightning as lt
from tqdm import tqdm
from matplotlib import pyplot as plt

class Diffusion(lt.LightningModule):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,volume_depth=16, img_size=256, c_in=1, c_out=1,encoder=None,attention=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().cuda()
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0).cuda()

        self.volume_depth = volume_depth
        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out,img_size=self.img_size,volume_depth=self.volume_depth,time_dim=256,encoder=encoder,attention=attention)
        self.ema_model = None
        self.c_in = c_in
        self.c_out = c_out
        
        

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    
    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].cuda()
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].cuda()
        Ɛ = torch.randn_like(x).cuda()
        return sqrt_alpha_hat * x.cuda() + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def plot_test_noising(self,x):
        x = nn.BatchNorm3d(1,device='cuda')(x.unsqueeze(1).unsqueeze(0).unsqueeze(0))
        x = x.squeeze()
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 11, 1)
        plt.imshow(x.detach().cpu().numpy())
        plt.title("t=0")
        plt.axis('off')
        for i in range(1, self.noise_steps,self.noise_steps // 10):
            t = torch.ones(x.shape[0], dtype=torch.long) * i
            x_t, _ = self.noise_images(x, t)
            x_t = (x_t.clamp(-1, 1) + 1) / 2
            x_t = (x_t * 255).type(torch.uint8)
            plt.subplot(1, 11, i // (self.noise_steps // 10) + 1)
            plt.imshow(x_t[0].squeeze().cpu().numpy(), cmap='gray')
            plt.title(f"t={i}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        #print uniques and their counts
        unique, counts = torch.unique(x_t[0].squeeze(), return_counts=True)
        print("Unique values and their counts:")
        for u, c in zip(unique, counts):
            print(f"Value: {u.item()}, Count: {c.item()}")
        
        
    
    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        labels = nn.BatchNorm3d(1,device='cuda')(labels.unsqueeze(1))
        labels = labels.squeeze()
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).cuda()
            for i in tqdm(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long()
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x).cuda()
                else:
                    noise = torch.zeros_like(x).cuda()
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def training_step(self, batch):
        ink_labels = batch['ink_label'].float().unsqueeze(1).cuda()
        ink_labels = nn.BatchNorm2d(1,device='cuda')(ink_labels)
        pergament = batch['scroll_segment'].float().unsqueeze(1).cuda()
        pergament = nn.BatchNorm3d(1,device='cuda')(pergament)
        t = self.sample_timesteps(len(ink_labels))
        x_t, noise = self.noise_images(ink_labels, t)
        pred_noise = self.model(x_t, t, pergament.squeeze())
        loss = nn.functional.mse_loss(pred_noise, noise)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ink_labels = batch['ink_label'].float().unsqueeze(1).cuda()
        ink_labels = nn.BatchNorm2d(1,device='cuda')(ink_labels)
        pergament = batch['scroll_segment'].float().unsqueeze(1).cuda()
        pergament = nn.BatchNorm3d(1,device='cuda')(pergament)
        t = self.sample_timesteps(len(ink_labels))
        x_t, noise = self.noise_images(ink_labels, t)
        pred_noise = self.model(x_t, t, pergament.squeeze())
        loss = nn.functional.mse_loss(pred_noise, noise)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        ink_labels = batch['ink_label'].float().unsqueeze(1).cuda()
        ink_labels = nn.BatchNorm2d(1,device='cuda')(ink_labels)
        pergament = batch['scroll_segment'].float().unsqueeze(1).cuda()
        pergament = nn.BatchNorm3d(1,device='cuda')(pergament)
        t = self.sample_timesteps(len(ink_labels))
        x_t, noise = self.noise_images(ink_labels, t)
        pred_noise = self.model(x_t, t, pergament.squeeze())
        loss = nn.functional.mse_loss(pred_noise, noise)
        self.log('test_loss', loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, eps=1e-5)
        return optimizer

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))

    def save(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        os.makedirs(model_cpkt_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_cpkt_path, model_ckpt))
    
    def __test__(self):
        scroll_segments = torch.rand(1,16,32,32).to(self.device)
        output = self.sample(use_ema=False, labels=scroll_segments,cfg_scale=0)
        print(f"Output shape: {output.shape}")
        return output

    def __test_after_training__(self, scroll_segments):
        self.cuda()
        output = self.sample(use_ema=False, labels=scroll_segments,cfg_scale=0)
        print(f"Output shape after training: {output.shape}")
        return output


