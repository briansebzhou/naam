import numpy as np
from naa import models as models

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torchvision.datasets as datasets
import os


data_transforms = {
    "train": transforms.Compose([transforms.ToTensor()]),
    "val": transforms.Compose([transforms.ToTensor()]),
    "test": transforms.Compose([transforms.ToTensor()]),
}

data_dir = "YOUR DATA DIR HERE" 
num_workers = {"train": 2, "val": 0, "test": 0}
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]
}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=num_workers[x])
    for x in ["train", "val", "test"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}
trainloader = dataloaders['train']

import random

# Load CIFAR10 dataset
transform = transforms.Compose([transforms.ToTensor()])

# Assuming 'net' is your neural network model
dt = 0.1
steps = 10
net = models.ContinuousRNN(input_size = int(64*64*3), dt = dt, steps = steps, device = device) # Replace with your network
#autoencoder = autoencoder.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# Function to generate patch locations
def generate_patch_locations(num_patches, patch_size, img_size):
    locations = []
    for _ in range(num_patches):
        i = random.randint(0, img_size - patch_size)
        j = random.randint(0, img_size - patch_size)
        locations.append((i, j))
    return locations

# Function to mask patches of the image
def mask_patches(img, locations, patch_size):
    masked_img = img.clone()
    for (i, j) in locations:
        masked_img[:, i:i+patch_size, j:j+patch_size] = 0
    return masked_img

# Training loop
num_epochs = 20  # Define the number of epochs
num_patches = 4  # Number of patches to mask
patch_size = 10  # Size of each patch
img_size = 64   # CIFAR10 images are 32x32
randomize_patches = True  # Set to False to keep patch locations same across the batch



# Training loop
for epoch in range(num_epochs):  # num_epochs is the number of epochs

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            # Generate patch locations for the batch
            patch_locations = generate_patch_locations(num_patches, patch_size, img_size)

            # Mask patches and move to device
            masked_inputs = [mask_patches(inp, patch_locations, patch_size) for inp in inputs]
            masked_inputs = torch.stack(masked_inputs).to(device)
            originals = inputs.to(device)

        # Forward + backward + optimize
        outputs = net(masked_inputs.view(16,-1))
        
        #with torch.no_grad():
            #masked_decoded = autoencoder.decoder(outputs)
        
        loss = criterion(outputs.view(16,3,64,64), inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
            
print('Finished Training')

