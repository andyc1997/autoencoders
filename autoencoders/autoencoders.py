from torch import nn
from torch.nn import functional as F


# Shallow autoencoder architecture
class AEobj(nn.Module):
    def __init__(self, hidden_dim=None, input_dim=None):
        super(AEobj, self).__init__()

        # ensure an undercomplete AE
        assert hidden_dim < input_dim
        assert hidden_dim > 0
        assert input_dim > 0

        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(input_dim, hidden_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Deep autoencoder architecture
class DeepAEobj(nn.Module):
    def __init__(self, hidden_dims=None, input_dim=None):
        super(DeepAEobj, self).__init__()

        # encoders
        self.encoder1 = Encoder(input_dim, hidden_dims[0])
        self.encoder2 = Encoder(hidden_dims[0], hidden_dims[1])
        self.encoder3 = Encoder(hidden_dims[1], hidden_dims[2])

        # decoders
        self.decoder1 = Decoder(input_dim, hidden_dims[0])
        self.decoder2 = Decoder(hidden_dims[0], hidden_dims[1])
        self.decoder3 = Decoder(hidden_dims[1], hidden_dims[2])

    def forward(self, x):
        # encode
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        # decode
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)
        return x


#region encoder function g
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x
#endregion


#region decoder function f
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x
#endregion
