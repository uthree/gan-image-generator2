import os
import sys

from trainer import StyleBasedGANTrainer
from dataset import ImageDataset
from PIL import Image
import numpy as np

model_path = 'model.pt'
test_result_dir = './tests'

trainer = StyleBasedGANTrainer.load(model_path)
print("Loaded model from disk.")

if len(sys.argv) > 1:
    num_images = int(sys.argv[1])
else:
    num_images = 1

images = trainer.generate_images(num_images)

if not os.path.exists(test_result_dir):
    os.mkdir(test_result_dir)
    
for i, image in enumerate(images):
    image = Image.fromarray(image)
    image = image.resize((512, 512))
    image.save(os.path.join(test_result_dir, '{}.jpg'.format(i)))