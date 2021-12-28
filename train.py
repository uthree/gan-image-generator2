import os
import sys
from trainer import StyleBasedGANTrainer
from dataset import ImageDataset

from augmentor import ImageDataAugmentor
from augmentations import *

model_path = 'model.pt'
if os.path.exists(model_path):
    trainer = StyleBasedGANTrainer.load(model_path)
    print("Loaded model from disk.")
else:
    trainer = StyleBasedGANTrainer(latent_dim=512, initial_channels=512, num_mapping_network_layers=8)
    print("Created new model.")

augmentor = ImageDataAugmentor()
augmentor.add_function(flip)
augmentor.add_function(contrast)
augmentor.add_function(random_roll)

dataset = ImageDataset(source_dir_pathes=sys.argv[1:], max_len=100)

trainer.train(dataset=dataset, initial_batch_size=32, num_epochs_per_resolution=100, max_resolution=1024, learning_rate=1e-4, save_path=model_path, results_dir_path='results/', augmentor=augmentor)