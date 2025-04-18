# %% Prepare data
import torchvision.datasets as datasets
import numpy as np

mnist_trainset = datasets.MNIST(root='./data', train=True, 
                                download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
train_set = mnist_trainset.data.numpy()
test_set = mnist_testset.data.numpy()
train_labels = mnist_trainset.targets.numpy()
test_labels = mnist_testset.targets.numpy()

# %% Import libraries
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset 
import matplotlib.pyplot as plt

# %% Check for gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# %% Show an image function
def imshow(img):
    plt.imshow(img, cmap='gray')
    plt.show()

# %% Show random 9 images
import random
random_range = random.sample(range(0, len(train_set)), 9)
j = 0
for i in random_range:
    plt.subplot(3,3,j+1)
    plt.imshow(train_set[i],cmap='gray')
    plt.title(train_labels[i])
    j += 1

# %% Dataset class
class MNISTDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.int64).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# %% split train set and dev set
X_train, X_dev, y_train, y_dev = train_test_split(train_set, train_labels, test_size=0.2, random_state=0)

#%% #train on training set only  
#train_loader = DataLoader(MNISTDataset(X_train, y_train), batch_size=64, shuffle=True)
#train on whole set
train_loader = DataLoader(MNISTDataset(train_set, train_labels)
                          , batch_size=64, shuffle=False,)
# %% Network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1) # 28x28x1 -> 26x26x16
        #pool -> 13x13x16
        self.conv2 = nn.Conv2d(16, 32, 3, 1) # 13x13x16 -> 11x11x32
        #pool -> 5x5x32
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5*5*32, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)
        self.relu = nn.ReLU()
    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.flatten(X)
        X = self.relu(self.fc1(X))
        X = self.relu(self.fc2(X))
        X = self.fc3(X)
        return X
#%% Load model
model = NeuralNetwork()
model.to(device)
#model.load_state_dict(torch.load('model.pth'))
#model.eval()

#%% Train the model
# Hyperparameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
losses = []

#%% 1000 epochs with batch_size=64
for epoch in range(100):
    for i, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(X_batch.unsqueeze(1).float())
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    if epoch % 1 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        losses.append(loss.item())

# %% Save the model
# torch.save(model.state_dict(), 'model.pth')

# %% Accuracy evaluation on dev set
from sklearn.metrics import accuracy_score
with torch.no_grad():
    y_pred = model(torch.tensor(X_dev, dtype=torch.float32).unsqueeze(1).to(device)).argmax(dim=1).cpu().numpy()
    print(f'Accuracy: {accuracy_score(y_dev, y_pred)}')

# %% Accuracy evaluation on test set
with torch.no_grad():
    y_pred = model(torch.tensor(test_set, dtype=torch.float32).unsqueeze(1).to(device)).argmax(dim=1).cpu().numpy()
    print(f'Accuracy: {accuracy_score(test_labels, y_pred)}')

# %% Plot the losses
x = [i*100 for i in range(len(losses))]
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(x, losses)

# %% Done
