import torch
from torch import nn

# convolutional autoencoder (CAE) architecture
class CAEobj(nn.Module):
    def __init__(self):
        super(CAEobj, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(2, 2), stride=2)
        self.flatten1 = nn.Flatten(1, 3) # flatten along dimension 1 when minibatch is used, dimension 0 is reserved for minibatch
        self.fc1 = nn.Linear(128*3*3, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(self.flatten1(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size=(2, 2), stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=2)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2) # use (3, 3) as kernel_size otherwise it results in 6x6x64
        self.unflatten1 = nn.Unflatten(1, torch.Size([128, 3, 3])) # unflatten along dimension 1 when minibatch is used, dimension 0 is reserved for minibatch
        self.fc1 = nn.Linear(10, 128*3*3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.unflatten1(x)
        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)
        return x
