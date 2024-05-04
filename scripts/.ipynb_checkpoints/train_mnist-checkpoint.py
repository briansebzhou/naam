import argparse
import na_utils
import models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == 'cuda', "No GPU!"

# load in MNIST
batch_size = 512
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#custom loss for caring strongly about mistakes (m >> 1)
class CustomMSELoss(nn.Module):
    def __init__(self, m=2):
        super(CustomMSELoss, self).__init__()
        self.m = m

    def forward(self, outputs, targets):
        loss = torch.pow(outputs - targets, self.m).mean()
        return loss
    
def train(epochs, T, dt, lr=1e-3, gamma=1, occlusion_size=2, p=3):

    #load MNIST encoder
    mnist_autoencoder = models.Autoencoder().to(device)  # Transfer model to the appropriate device (CPU or GPU)
    
    # load autoencoder here
     mnist_autoencoder.load_state_dict(torch.load('PATH_TO_YOUR_AUTOENCODER', map_location=torch.device('cuda')))   

    # freeze autoencoder weights
    for param in mnist_autoencoder.parameters():
        param.requires_grad = False

    steps = round(T/dt)

    #define neuron-astrocyte model and loss function
    model = models.NeuronAstrocyteNetwork(25,25, dt = dt, steps = steps, device = device)
    criterion = CustomMSELoss(m = 4)
    optimizer = optim.Adam(model.parameters(), lr = lr,weight_decay = 1e-4)
    scheduler = ExponentialLR(optimizer, gamma = gamma)

    # Step 4: Define the training loop
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            with torch.no_grad():      
                data_encoded = mnist_autoencoder.encoder(data.view(data.size(0),-1))
                corrupted_data_encoded,corruption_mask = na_utils.zero_out_elements(data_encoded,zero_prob = 0.1,device = device)

            optimizer.zero_grad()

            outputs = model(corrupted_data_encoded)
            model_decoded = mnist_autoencoder.decoder(outputs)

            loss = criterion(model_decoded, data.view(data.size(0),-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()


            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Log Loss: {np.log(loss.item()):.6f}")
        scheduler.step()

    return model




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuronAstrocyteNetwork with specified T and dt")
    parser.add_argument('--T', type=float, required=True, help='Specify the value of T')
    parser.add_argument('--dt', type=float, required=True, help='Specify the value of dt')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train for')
    args = parser.parse_args()
    save_path = ""

    trained_model = train(epochs = args.epochs,T=args.T, dt=args.dt)

    
    torch.save(trained_model.state_dict(), save_path)
    print('Training completed. Model saved.')