from math import exp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import multiprocessing
import os
import numpy as np
from PIL import Image

from model import StyleBasedGAN
from dataset import ImageDataset

class StyleBasedGANTrainer:
    def __init__(self, latent_dim, initial_channels=512, num_mapping_network_layers=8):
        self.stylegan = StyleBasedGAN(latent_dim, initial_channels, num_mapping_network_layers)
        self.g = self.stylegan.generator
        self.d = self.stylegan.disccriminator
        self.m = self.stylegan.mapping_network
        self.resolution = 4
    
    @classmethod
    def load(self, path='model.pt'):
        self = torch.load(path)
        self.resolution = 4 * 2 ** (len(self.g.layers))
        return self
    
    def save(self, path='model.pt'):
        torch.save(self, path)
    
    def train(self, dataset: ImageDataset, initial_batch_size=64, num_epochs_per_resolution=1, max_resolution=1024, learning_rate=1e-4, save_path='model.pt', results_dir_path='results/', checkpoints_dir_path='checkpoints/', augmentor=None):
        bs = initial_batch_size
        if not os.path.exists(results_dir_path):
            os.mkdir(results_dir_path)
        while self.resolution <= max_resolution:
            self.resolution = int(4 * 2 ** (len(self.g.layers)))
            self.train_resolution(dataset, batch_size=bs, num_epochs=num_epochs_per_resolution, learning_rate=learning_rate, save_path=save_path, results_dir_path=results_dir_path, checkpoints_dir_path=checkpoints_dir_path, augmentor=augmentor)
            channels = self.g.last_channels // 2
            if channels <= 8:
                channels = 8
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.resolution == max_resolution:
                print(f"Finished training.")
                return
            self.stylegan.add_layer(channels)
            self.resolution = int(4 * 2 ** (len(self.g.layers)))
            
            bs = bs // 2
            if bs < 2:
                bs = 2
            print(f"Added layer with {channels} channels. now {len(self.g.layers)} layers. batch size: {bs}.")

        
    def train_resolution(self, dataset: ImageDataset, batch_size=1, num_epochs=1, learning_rate=1e-4, save_path='model.pt', results_dir_path='results/', checkpoints_dir_path='checkpoints/', augmentor=None):
        if not os.path.exists(checkpoints_dir_path):
            os.mkdir(checkpoints_dir_path)
        
        if augmentor is None:
            augmentor = lambda x: x
        
        dataset.set_size(self.resolution)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_cpus = multiprocessing.cpu_count() - 1
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_cpus)
        progress_total = (len(dataset) // batch_size + 1) * num_epochs
        
        g = self.g.to(device)
        d = self.d.to(device)
        m = self.m.to(device)
        
        optimizer_g = optim.Adam(g.parameters(), lr=learning_rate)
        optimizer_m = optim.Adam(m.parameters(), lr=learning_rate)
        optimizer_d = optim.Adam(d.parameters(), lr=learning_rate)
        
        MSE = nn.MSELoss()
        
        print(f"Started training with {num_cpus} workers in resolution {self.resolution}x.")
        bar = tqdm(total=progress_total)
        bar_now = 0
        for epoch in range(num_epochs):
            for i, (image) in enumerate(dataloader):
                try:
                    N = image.shape[0]
                    
                    image = image.to(device)
                    z = torch.randn(N, self.stylegan.latent_dim).to(device)
    
                    alpha = bar_now / progress_total
                    g.alpha = alpha
                    d.alpha = alpha
                    
                    # Train Generator
                    m.zero_grad()
                    g.zero_grad()
                    
                    # get mapping network output
                    z = m(z)
                    
                    # generate image
                    fake_image = g(z)
                    g_loss = MSE(d(fake_image), torch.ones(N, 1).to(device)) 
                    
                    g_loss.backward()
                    optimizer_g.step()
                    optimizer_m.step()
                    

                    # Train Discriminator
                    d.zero_grad()
                    fake_image = augmentor(fake_image.detach())
                    real_image = augmentor(image).detach()

                    d_loss = (MSE(d(real_image), torch.ones(N, 1).to(device)) + MSE(d(fake_image), torch.zeros(N, 1).to(device))) / 2
                    
                    d_loss.backward()
                    optimizer_d.step()
                    
                    description = f"Epoch {epoch + 1}/{num_epochs}, Batch:{i} GLoss: {g_loss.item():.4f}, DLoss: {d_loss.item():.4f}, Alpha: {alpha:.4f}"
                    bar.set_description(description)
                    if g_loss + d_loss > 80 and alpha > 0.1:
                        tqdm.write("Losses are too high. Stopping training.")
                        return
                    
                    if i % 1000 == 0:
                        self.save(save_path)
                        tqdm.write(description)
                        img = fake_image[0].detach().cpu().numpy() * 127.5 + 127.5
                        img = img.astype(np.uint8)
                        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
                        img.save(os.path.join(results_dir_path, f"{epoch}_{i}.png"))
                        if epoch % 10 == 0:
                            self.save(os.path.join(checkpoints_dir_path, f"{self.resolution}x_{epoch}.pt"))
                            tqdm.write("Checkpoint saved.")
                except Exception as e:
                    tqdm.write(f"Error occured. Skipping batch.\n{e}")
                finally:
                    bar.update(1)
                    bar_now += 1
                    
    @torch.no_grad()
    def generate_images(self, num_images):
        results = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = self.m.to(device)
        g = self.g.to(device)
        for i in tqdm(range(num_images)):
            z = torch.randn(1, self.stylegan.latent_dim).to(device)
            z = m(z)
            img = g(z)
            img = img.detach().cpu().numpy()[0]
            img = np.transpose(img, (1, 2, 0))
            img = img * 127.5 + 127.5
            img = img.astype(np.uint8)
            results.append(img)
        return results