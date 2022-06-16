#!/usr/bin/env python
# coding: utf-8

# # Training LeNet using MNIST and Joey

# In this notebook, we will construct and train LeNet using Joey, data from MNIST and the SGD with momentum PyTorch optimizer.

# Let's start with importing the prerequisites:

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joey as ml
import numpy as np
import matplotlib.pyplot as plt
from devito import logger

# In order to speed up processing, we'll not print performance messages coming from Devito.

# In[2]:


logger.set_log_noperf()


# `create_lenet()` returns a `Net` instance representing LeNet.

# In[3]:


def create_lenet():
    # Six 3x3 filters, activation RELU
    layer1 = ml.Conv(kernel_size=(6, 3, 3),
                     input_size=(batch_size, 1, 32, 32),
                     activation=ml.activation.ReLU())
    # Max 2x2 subsampling
    layer2 = ml.MaxPooling(kernel_size=(2, 2),
                           input_size=(batch_size, 6, 30, 30),
                           stride=(2, 2))
    # Sixteen 3x3 filters, activation RELU
    layer3 = ml.Conv(kernel_size=(16, 3, 3),
                     input_size=(batch_size, 6, 15, 15),
                     activation=ml.activation.ReLU())
    # Max 2x2 subsampling
    layer4 = ml.MaxPooling(kernel_size=(2, 2),
                           input_size=(batch_size, 16, 13, 13),
                           stride=(2, 2),
                           strict_stride_check=False)
    # Full connection (16 * 6 * 6 -> 120), activation RELU
    layer5 = ml.FullyConnected(weight_size=(120, 576),
                               input_size=(576, batch_size),
                               activation=ml.activation.ReLU())
    # Full connection (120 -> 84), activation RELU
    layer6 = ml.FullyConnected(weight_size=(84, 120),
                               input_size=(120, batch_size),
                               activation=ml.activation.ReLU())
    # Full connection (84 -> 10), output layer
    layer7 = ml.FullyConnectedSoftmax(weight_size=(10, 84),
                                      input_size=(84, batch_size))
    # Flattening layer necessary between layer 4 and 5
    layer_flat = ml.Flat(input_size=(batch_size, 16, 6, 6))

    layers = [layer1, layer2, layer3, layer4,
              layer_flat, layer5, layer6, layer7]

    return (ml.Net(layers), layers)


# A proper training iteration is carried out in `train()`. Note that we pass a PyTorch optimizer to `net.backward()`. Joey will take care to use it for updating weights appropriately.

# In[4]:


def train(net, input_data, expected_results, pytorch_optimizer):
    outputs = net.forward(input_data)

    def loss_grad(layer, expected):
        gradients = []

        for b in range(len(expected)):
            row = []

            for i in range(10):
                result = layer.result.data[i, b]
                if i == expected[b]:
                    result -= 1
                row.append(result)

            gradients.append(row)

        return gradients

    net.backward(expected_results, loss_grad, pytorch_optimizer)


# In this example, every batch will consist of 4 images and the training session will be capped at 100 iterations.

# In[5]:


batch_size = 4
iterations = 100

# Before starting training, we need to download MNIST data using PyTorch.

# In[6]:


transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])
trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Afterwards, let's instantiate Joey's LeNet along with the SGD with momentum PyTorch optimizer.

# In[7]:


devito_net, devito_layers = create_lenet()
optimizer = optim.SGD(devito_net.pytorch_parameters, lr=0.001, momentum=0.9)

# We're almost ready! The last thing to do is saving our original parameters as they will be required for making later comparisons with PyTorch.

# In[8]:


layer1_kernel = torch.tensor(devito_layers[0].kernel.data)
layer1_bias = torch.tensor(devito_layers[0].bias.data)
layer3_kernel = torch.tensor(devito_layers[2].kernel.data)
layer3_bias = torch.tensor(devito_layers[2].bias.data)
layer5_kernel = torch.tensor(devito_layers[5].kernel.data)
layer5_bias = torch.tensor(devito_layers[5].bias.data)
layer6_kernel = torch.tensor(devito_layers[6].kernel.data)
layer6_bias = torch.tensor(devito_layers[6].bias.data)
layer7_kernel = torch.tensor(devito_layers[7].kernel.data)
layer7_bias = torch.tensor(devito_layers[7].bias.data)

# We can start the Joey training session now.

# In[9]:


for i, data in enumerate(trainloader, 0):
    images, labels = data
    images.double()

    train(devito_net, images, labels, optimizer)

    if i == iterations - 1:
        break


# Afterwards, let's create a PyTorch equivalent of Joey's LeNet, train it using the same initial weights and data and compare the results.

# In[10]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[11]:


net = Net()
net.double()

with torch.no_grad():
    net.conv1.weight[:] = layer1_kernel
    net.conv1.bias[:] = layer1_bias
    net.conv2.weight[:] = layer3_kernel
    net.conv2.bias[:] = layer3_bias
    net.fc1.weight[:] = layer5_kernel
    net.fc1.bias[:] = layer5_bias
    net.fc2.weight[:] = layer6_kernel
    net.fc2.bias[:] = layer6_bias
    net.fc3.weight[:] = layer7_kernel
    net.fc3.bias[:] = layer7_bias

# In[12]:


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
for i, data in enumerate(trainloader, 0):
    images, labels = data
    optimizer.zero_grad()
    outputs = net(images.double())
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if i == iterations - 1:
        break

# In[13]:


layers = [devito_layers[0], devito_layers[2], devito_layers[5], devito_layers[6], devito_layers[7]]
pytorch_layers = [net.conv1, net.conv2, net.fc1, net.fc2, net.fc3]

max_error = 0
index = -1

for i in range(5):
    kernel = layers[i].kernel.data
    pytorch_kernel = pytorch_layers[i].weight.detach().numpy()

    kernel_error = abs(kernel - pytorch_kernel) / abs(pytorch_kernel)

    bias = layers[i].bias.data
    pytorch_bias = pytorch_layers[i].bias.detach().numpy()

    bias_error = abs(bias - pytorch_bias) / abs(pytorch_bias)

    error = max(np.nanmax(kernel_error), np.nanmax(bias_error))
    print('layers[' + str(i) + '] maximum relative error: ' + str(error))

    if error > max_error:
        max_error = error
        index = i

print()
print('Maximum relative error is in layers[' + str(index) + ']: ' + str(max_error))

# As we can see, the maximum relative error is low enough to consider the training session in Joey numerically correct.