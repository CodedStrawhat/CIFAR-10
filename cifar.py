#CIFAR 10 classification using CNN

#Loading required libraries
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as td

#cuda will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epoch = 10 
batch_size = 5
alpha = 0.01

#applying transformations to input data (converting np array to tensor and normalizing values)
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#splitting train and test dataset
train_dataset = td.CIFAR10(root = './sample_data',train = True,download = True,transform = transform)
test_dataset = td.CIFAR10(root = './sample_data',train = False,transform = transform)

n_test = len(test_dataset)
train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
test_loader = DataLoader(dataset = test_dataset,batch_size = n_test,shuffle = False)

#building the module
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 =  nn.Conv2d(3,6,5,padding = 'same') #1st conv layer with 3 input channels, 6 output channels and 5x5 kernel size and same padding
    self.bn1 = nn.BatchNorm2d(6) 
    self.pool = nn.MaxPool2d(2,2) #pooling will reduce size by 2
    self.conv2 = nn.Conv2d(6,16,5,padding = 'same') #2nd conv layer with 6 input channels 16 output channels and 5x5 kerelsize and same padding
    self.bn2 = nn.BatchNorm2d(16)
    self.l1 =nn.Linear(16*8*8,120) #1st layer of neural network
    self.bn3 = nn.BatchNorm1d(120)
    self.l2 = nn.Linear(120,84)   #2nd layer of neural network
    self.bn4 = nn.BatchNorm1d(84)
    self.l3 = nn.Linear(84,10)   #output layer
  def forward(self,x):
    x = self.pool(torch.relu(self.bn1((self.conv1(x)))))  #1st filter
    x = self.pool(torch.relu(self.bn2(self.conv2(x))))    #2nd filter
    x = x.view(-1,16*8*8)                                 #flattening
    x = torch.relu(self.bn3(self.l1(x)))                  #1st layer
    x = torch.relu(self.bn4(self.l2(x)))                  #2nd layer
    x = self.l3(x)                                        #output without softmax
    return x
model = CNN().to(device)
criterion = nn.CrossEntropyLoss() #loss function includes softmax
optimizer = torch.optim.Adam(model.parameters(),lr = alpha)  #using adam optimizer
steps = len(train_loader) #number of steps = [number of samples/batch_size]
for epoch in range(num_epoch):
  for i,(images,labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    #forward pass
    y_pred = model(images)                  
    loss = criterion(y_pred,labels)   
    #backward pass      
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(i+1)%2000 == 0:
      print (f'Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{steps}], Loss: {loss.item():.4f}')

#evaluating model with test
with torch.no_grad():
  n_correct = 0
  data = iter(test_loader)
  images,labels = data.next()
  images = images.to(device)
  labels = labels.to(device)
  y_pred = model(images)
  _,y_pred = torch.max(y_pred,1)
  n_correct = (y_pred==labels).sum().item()


acc = 100.0 * n_correct / n_test
print(f'Accuracy of the network: {acc} %')
