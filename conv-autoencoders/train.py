import numpy as np
import idx2numpy
import torch

from model import CAEobj
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

## MNIST


#region load images into tensor
X_tr = idx2numpy.convert_from_file(r'train-images.idx3-ubyte')
X_tst = idx2numpy.convert_from_file(r't10k-images.idx3-ubyte')

(N_tr, nx, ny) = X_tr.shape
(N_tst, _, _) = X_tst.shape

# add single channe
X_tr = np.expand_dims(X_tr, axis=1)
X_tst = np.expand_dims(X_tst, axis=1)

# normalize into [0, 1]
X_tr = torch.Tensor(X_tr/255.0)
X_tst = torch.Tensor(X_tst/255.0)

dataset_tr, dataset_tst = TensorDataset(X_tr), TensorDataset(X_tst)

batch_size = 64
dataloader_tr = DataLoader(X_tr, batch_size=batch_size)
dataloader_tst = DataLoader(X_tst, batch_size=batch_size)
#endregion


#region device, model and optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = CAEobj().to(device)
print(model)

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-6)

#endregion


#region training
max_epoch = 100
loss_values = []
loss_values_tst = []
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
        running_loss.append(loss.item()) # extract loss' value as a python float
        if (i % 100 == 0) and (i > 0):
            print(f'Epoch: {epoch} \t Iteration: {i} \t Loss: {np.round(np.mean(running_loss), decimals=6)}')

    loss_values.append(np.mean(running_loss))

    running_loss = []
    model.eval()
    for i, x in enumerate(dataloader_tst, 0):
        x_pred = model(x.to(device))
        loss = criterion(x_pred, x.to(device))
        running_loss.append(loss.item())

    loss_values_tst.append(np.mean(running_loss))
#endregion


#region plot loss
from matplotlib import pyplot as plt

plt.plot(np.array(loss_values), label='training loss')
plt.plot(np.array(loss_values_tst), label='test loss')

plt.xlabel('Epoch number')
plt.ylabel('Loss (MSE)')
plt.title('Loss curve')
plt.legend(loc='upper right')
#endregion


#region save model
model_ver = r'.\CAE_model1.pth' # .pt or .pth means a pytorch object
torch.save(model.state_dict(), model_ver)

#endregion


#region selected example
fig, axs = plt.subplots(1, 2)

# source image
idx_select = 90
img_src = X_tr[idx_select, :].reshape((nx, ny)) # source image, reshape to 1D

axs[0].imshow(img_src, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('Source image')
axs[0].axis('off')

# reconstructed image
img_recon = torch.Tensor(img_src.reshape((1, 1, nx, ny))) # reconstructed image
img_recon = model(img_recon.to(device)) # prediction
img_recon = img_recon.detach().cpu().numpy() # detach it from computation graph, move from gpu to cpu and convert to numpy
img_recon = img_recon.reshape((nx, ny)) # reshape to 2D

axs[1].imshow(img_recon, cmap='gray', vmin=0, vmax=1)
axs[1].set_title('Reconstruction')
axs[1].axis('off')

#endregion


#region latent space
# load model
device = torch.device('cpu')
model_load = CAEobj()
model_load.load_state_dict(torch.load('CAE_model1.pth', map_location=device))

# compute latent space
latent_space = model_load.encoder(X_tr.to(device))
latent_space = latent_space.detach().cpu().numpy()

path = r'.\latent_space_MNIST.pth' # .pt or .pth means a pytorch object
torch.save(latent_space, path)

# latent space traversal
col_select = 9
h_min, h_max = np.min(latent_space[:, col_select]), np.max(latent_space[:, col_select])
img_latent = latent_space[50, :]

num_plots = 10
fig, axs = plt.subplots(1, num_plots)
for i, h in enumerate(np.linspace(h_min, h_max, num=num_plots)):
    img_latent[col_select] = h
    img_recon = model_load.decoder(torch.Tensor(img_latent.reshape((1, 10))))
    img_recon = img_recon.reshape((nx, ny))

    axs[i].imshow(img_recon.detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axs[i].set_title('Reconstruction')
    axs[i].axis('off')

#endregion