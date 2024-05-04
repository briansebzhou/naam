import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.manual_seed(SEED)


def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")


def create_model():
    autoencoder = Autoencoder().to(device)
    print_model(autoencoder.encoder, autoencoder.decoder)
    if device.type == 'cuda':
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Custom round function
class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def round_with_gradient(x):
    return RoundWithGradient.apply(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.GELU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.GELU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.GELU(),
            nn.Flatten(),                                        # Flatten the output for the linear layer
            nn.Linear(48*4*4, 48*4*4),                             # Linear layer to get to the latent space
            nn.Tanh()                                            # Tanh activation to scale values to [-1, 1]
        )
        self.decoder = nn.Sequential(
            nn.Linear(48*4*4, 48*4*4),                             # Linear layer to go back from the latent space
            nn.GELU(),
            nn.Unflatten(1, (48, 4, 4)),                         # Unflatten to get back to the conv shape
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.GELU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.GELU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        #latent = round_with_gradient(encoded)
        decoded = self.decoder(encoded)
        return decoded


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = torchvision.datasets.CIFAR10(root='YOUR DATA FOLDER HERE', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='YOUR DATA FOLDER HERE', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # To perform validation only, set valid=True
    valid = False

    autoencoder = create_model()

    if valid:
        print("Loading checkpoint...")
        autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(16)))
        imshow(torchvision.utils.make_grid(images))

        images = images.to(device)
        decoded_imgs = autoencoder(images)[1]
        imshow(torchvision.utils.make_grid(decoded_imgs.data))

    else:
        criterion = nn.BCELoss().to(device)
        optimizer = optim.Adam(autoencoder.parameters(),lr = 2e-4)

        max_epochs = 50

        for epoch in range(max_epochs):
            running_loss = 0.0
            for i, (inputs, _) in enumerate(trainloader, 0):
                inputs = inputs.to(device)
                outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, max_epochs + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
        
        # write your code to save autoencoder here

if __name__ == "__main__":
    main()

