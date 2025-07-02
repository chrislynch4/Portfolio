# Chris Lynch
# Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import pymc3 as pm
import theano.tensor as T
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# GPU or CPU?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading and Normalizing CIFAR10
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*2*2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.drop = nn.Dropout(0.5)
        self.batch1 = nn.BatchNorm2d(6)
        self.batch2 = nn.BatchNorm2d(16)
        self.batch3 = nn.BatchNorm2d(32)
        self.batch4 = nn.BatchNorm2d(64)

    def forward(self, x):
        # Conv Layer 1
        x = self.conv1(x) # 3x32x32->6x28x28
        x = self.batch1(x) 
        x = F.relu(x) 
        x = self.conv2(x) # 6x28x28->16x24x24
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool(x) # 16x24x24->16x12x12
        # Conv Layer 2
        x = self.conv3(x) # 16x12x12->32x8x8
        x = self.batch3(x) 
        x = F.relu(x) 
        x = self.conv4(x) # 32x8x8->64x4x4
        x = self.batch4(x)
        x = F.relu(x)
        x = self.pool(x) # 64x4x4->64x2x2
        
        # FLattening
        x = x.view(-1, 64*2*2)
        
        # Linear Layers
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

net = Net()

# Define a Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the Network
num_epochs = 10
train_epoch_loss = [None for x in range(num_epochs)]
test_epoch_acc = [None for x in range(num_epochs)]

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0; train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        train_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    
    train_epoch_loss[epoch] = train_loss/i
    
    # Let us look at how the network performs on the whole dataset.
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_epoch_acc[epoch] = 100*(correct/total)
    
print('Finished Training')
print(train_epoch_loss)
print(test_epoch_acc)

# Save the Trained Model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# Test the Network on the Test Data
dataiter = iter(testloader)
images, labels = dataiter.next()

# Print Images
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# load back in the saved model
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# Let us look at how the network performs on the whole dataset.
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Large Network Train Loss per Epochs')
plt.plot(train_epoch_loss)
plt.show()

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Large Network Test Accuracy per Epochs')
plt.plot(test_epoch_acc)
plt.show()
