import idx2numpy
import numpy as np
import torch

from autoencoders import AEobj

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

#region load images into tensor
X_tr = idx2numpy.convert_from_file(r'train-images.idx3-ubyte')
(N_tr, nx, ny) = X_tr.shape

# [avoid] UserWarning: The given NumPy array is not writeable,
# and PyTorch does not support non-writeable tensors
X_tr = np.array(X_tr.reshape(N_tr, nx*ny))

# normalize into [0, 1]
X_tr = torch.Tensor(X_tr/255.0)
dataset_tr = TensorDataset(X_tr)

batch_size = 64
dataloader_tr = DataLoader(X_tr, batch_size=batch_size)
#endregion


#region device, model and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

hidden_dims = [392, 196, 98]
input_dims = [nx*ny, hidden_dims[0], hidden_dims[1]]

models = [AEobj(hidden_dim=hidden_dims[i], input_dim=input_dims[i]).to(device) for i in range(len(hidden_dims))]
models = nn.ModuleList(models)
optimizers = [optim.Adam(models[i].parameters(), lr=1e-6) for i in range(len(hidden_dims))]


def train(model=None, optimizer=None, criterion=None, dataloader_tr=dataloader_tr, max_epoch=50):

    for epoch in range(max_epoch):
        running_loss = []
        model.train()

        for i, x in enumerate(dataloader_tr, 0):
            # sets gradients of all model parameters to zero.
            optimizer.zero_grad()

            # forward pass, backward and parameters update
            x_pred = model(x.to(device))
            loss = criterion(x_pred, x.to(device))
            loss.backward()
            optimizer.step()

            # print loss
            running_loss.append(loss.item())  # extract loss' value as a python float
            if (i % 100 == 0) and (i > 0):
                print(f'Hidden layer: {k} \t Epoch: {epoch} \t Iteration: {i} \t '
                      f'Loss: {np.round(np.mean(running_loss), decimals=6)}')


for k in range(len(hidden_dims)):
    #region update input data
    if k > 0:
        models[k-1].requires_grad_(False) # avoid previous model parameters being updated in the next loop
        X_tr = models[k-1].encoder(X_tr.detach().to(device))
        dataloader_tr = DataLoader(X_tr, batch_size=batch_size)
    #endreigon

    model, optimizer = models[k], optimizers[k]
    criterion = nn.MSELoss(reduction='mean')
    train(model, optimizer, criterion, dataloader_tr, max_epoch=50)

#endregion


#region save models
for k in range(len(hidden_dims)):
    model_ver = r'.\AE_layer{k}_model.pth'  # .pt or .pth means a pytorch object
    torch.save(models[k].state_dict(), model_ver.format(k=str(k)))

#endregion