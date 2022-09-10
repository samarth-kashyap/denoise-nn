import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torchvision import datasets
import torchvision.transforms as transforms

from networks import autoencoders as AE
#-----------------------------------------------------------------------------
# Setting device to CUDA if gpu is available, else setting device to CPU
# GPU availability can be checked using torch.cuda.is_available() function call
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

transform = transforms.ToTensor() # convert data to torch.FloatTensor
#-----------------------------------------------------------------------------
# Using a wrapper to create custom dataset
class DenoisingDataset(Dataset):
    def __init__(self, ds_noisy, ds_target):
        self.ds_noisy = ds_noisy
        self.ds_target = ds_target
        
    def __getitem__(self, index):
        xA = self.ds_noisy[index]
        xB = self.ds_target[index]
        return xA, xB
    
    def __len__(self):
        return len(self.ds_noisy)


# Creating a custom npy loader
def npy_loader(path):
    sample = torch.from_numpy(np.load(path).astype('float32'))
    return sample
#-------------------- LOADING TRAINING AND TESTING DATA ---------------------
train_dataset_noisy = datasets.DatasetFolder(
    root='train-set/noisy',
    loader=npy_loader,
    extensions=['.npy']
)

train_dataset_target = datasets.DatasetFolder(
     root='train-set/target',
     loader=npy_loader,
     extensions=['.npy']
)

test_dataset_noisy = datasets.DatasetFolder(
    root='test-set/noisy',
    loader=npy_loader,
    extensions=['.npy']
)

test_dataset_target = datasets.DatasetFolder(
     root='test-set/target',
     loader=npy_loader,
     extensions=['.npy']
)
#-----------------------------------------------------------------------------
train_dataset = DenoisingDataset(train_dataset_noisy, train_dataset_target)
test_dataset = DenoisingDataset(test_dataset_noisy, test_dataset_target)

# Create training and test dataloaders
num_workers = 0  # number of subprocesses to use for data loading
batch_size = 30  # how many samples per batch to load
n_epochs = 20    # number of epochs to train the model
num_layers = 3   # number of layers in autoencoder
input_dim = 500  # dimension of loaded dataset
encod_dim = 50   # encoding dimension

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=True)

# initialize the NN
model = AE.LinearAutoencoder(input_dim, encod_dim, num_layers=num_layers)
model = model.to(device)
criterion = nn.MSELoss() # specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # specify optimizer function

os.system('ls train-set/metadata*.pkl > pkl_name')
with open('pkl_name', 'r') as f:
    metadata_file = f.read().splitlines()
fsuffix = metadata_file[0].split('/')[-1][9:-4]
os.system('rm pkl_name')


for epoch in range(1, n_epochs+1):
    train_loss = 0.0  # monitoring training loss
    test_loss = 0.0   # monitoring testing loss
    
    len_testloader = 0
    # main training loop
    for data in train_loader:
        # _ stands in for labels, here
        noisy, target = data
        noisy = noisy[0]
        target = target[0]

        tnoisy, ttarget = next(enumerate(test_loader))[1]
        tnoisy = tnoisy[0]
        ttarget = ttarget[0]
        tnoisy = tnoisy.to(device)
        ttarget = ttarget.to(device)
        toutput = model(tnoisy)
        tloss = criterion(toutput, ttarget)
        test_loss += tloss.item()*tnoisy.size(0)
        len_testloader += 1

        # converting to device array
        noisy = noisy.to(device)
        target = target.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(noisy)

        # calculate the loss
        loss = criterion(outputs, target)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        # update running training loss
        train_loss += loss.item()*noisy.size(0)
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    test_loss = test_loss/len_testloader
    print('Epoch: {:02d}  Training Loss: {:.6f}  Testing Loss: {:.6f}'\
          .format(epoch, train_loss, test_loss))
    
    if epoch > 14:
        torch.save(model.state_dict(), f'train-set/model-{epoch}-{fsuffix}.pth')
