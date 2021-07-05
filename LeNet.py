import torch 
import matplotlib.pyplot as plt
import numpy as np


#Data set=CIFAR10

import torchvision
import torchvision.transforms as transforms
trainset=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transforms.ToTensor())
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import torch.nn as nn
class LeNet(nn.Module):
  def __init__(self):
    super(LeNet,self).__init__()
    self.cnn_model=nn.Sequential(
        nn.Conv2d(3,6,5), # we are taking (N,3,32,32) and output for this is (N,6,28,28)
        nn.Tanh(),
        nn.AvgPool2d(2,stride=2),   #(N,6,28,28) and output is (N,6,14,14)
        nn.Conv2d(6,16,5),  #(N,6,14,14) and output is (N,16,10,10)
        nn.Tanh(),
        nn.AvgPool2d(2,stride=2) #(N,16,10,10) and output is (N, 16, 5,5)
        )
    self.fc_model=nn.Sequential(
        nn.Linear(400,120),   #(N,400) to (N,120)
        nn.Tanh(),
        nn.Linear(120,84), #(N,120) to (N, 84)
        nn.Tanh(),
        nn.Linear(84,10) #(N,84) to (N,10)

    )
  def forword(self,x):
    print(x.shape)
    x=self.cnn_model(x)
    print(x.shape)
    x=x.view(x.size(0),-1)
    print(x.shape)
    x=self.fc_model(x)
    print(x.shape)
    return x
net1= LeNet()
out=net1.forword(images)
batch_size = 128
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def evaluation(dataloader):
    total, correct = 0, 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net1.forword(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return 100 * correct / total
import torch.optim as optim
net1 = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(net1.parameters())
%%time
max_epochs = 16

for epoch in range(max_epochs):

    for i, data in enumerate(trainloader, 0):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        opt.zero_grad()

        outputs = net1.forword(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        
    print('Epoch: %d/%d' % (epoch, max_epochs))
print('Test acc: %0.2f, Train acc: %0.2f' % (evaluation(testloader), evaluation(trainloader)))
