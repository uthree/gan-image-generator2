import os

from trainer import StyleBasedGANTrainer
from dataset import ImageDataset

model_path = 'model.pt'
if os.path.exists(model_path):
    trainer = StyleBasedGANTrainer.load(model_path)
    print("Loaded model from disk.")
else:
    trainer = StyleBasedGANTrainer(latent_dim=512, initial_channels=512, num_mapping_network_layers=8)
    print("Created new model.")

dataset = ImageDataset(source_dir_pathes=[
    "/mnt/d/local-develop/lineart2image_data_generator/colorized_1024x",
    #"/mnt/d/local-develop/lineart2image_data_generator/colorized/",
    #"/mnt/d/local-develop/AnimeIconGenerator128x_v3/small_dataset128x",
],
                       max_len=100000,
)

trainer.train(dataset=dataset, initial_batch_size=32, num_epochs_per_resolution=50, max_resolution=1024, learning_rate=1e-4, save_path=model_path, results_dir_path='results/')