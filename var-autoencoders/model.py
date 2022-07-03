import torch
from torch import nn
from torch.nn import functional as F

# variational autoencoder (VAE) architecture
class VAEobj(nn.Module):
    def __init__(self, latent_dim):
        super(VAEobj, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparamterize(self, mean, logsig):
        eps = torch.normal()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#region encoder q(latent|data)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        # check latent dimension
        assert latent_dim > 0
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        self.flatten1 = nn.Flatten(start_dim=1, end_dim=3)
        self.fc1 = nn.Linear(64*6*6, 2*latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten1(x)
        x = self.fc1(x)
        return torch.split(x, 2, dim=1)

#endregion


#region decoder p(data|latent)
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(latent_dim, 7*7*32)
        self.unflatten1 = nn.Unflatten(1, (7, 7, 32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.unflatten1(x)
        return x
#endregion