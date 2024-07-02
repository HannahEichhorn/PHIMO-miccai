""" Base code from
https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118"""


import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import hyperlight as hl


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x    # return x for visualization


class HyperNet(nn.Module):

    def __init__(self, mainnet, input_dim=1, hidden_sizes=None):
        super().__init__()
        self.hidden_sizes = (hidden_sizes
                             if hidden_sizes is not None
                             else [16, 64, 128])
        self.input_dim = input_dim

        # Use HyperLight convenience functions to select relevant modules
        modules = hl.find_modules_of_type(
            mainnet, [nn.Conv2d, nn.Linear]
        )
        self.mainnet = hl.hypernetize(mainnet, modules=modules)
        parameter_shapes = self.mainnet.external_shapes()

        self.hypernet = hl.HyperNet(
            input_shapes={'h': (self.input_dim,)},
            output_shapes=parameter_shapes,
            hidden_sizes=self.hidden_sizes,
        )
        #self.apply(self.weight_init)

    def weight_init(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, hyper_input, main_input):
        parameters = self.hypernet(h=hyper_input)

        with self.mainnet.using_externals(parameters):
            prediction = self.mainnet(main_input)

        return prediction



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor()
)

print(train_data)


plt.imshow(train_data.data[0], cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()


loaders = {
    'train': torch.utils.data.DataLoader(train_data,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=1),

    'test': torch.utils.data.DataLoader(test_data,
                                        batch_size=100,
                                        shuffle=True,
                                        num_workers=1),
}

cnn_direct = CNN()
print(cnn_direct)

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(cnn_direct.parameters(), lr = 0.01)

num_epochs = 10


def train(num_epochs, cnn_direct, loaders):
    cnn_direct.train()

    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y

            output = cnn_direct(b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()  # apply gradients
            optimizer.step()

            if (i + 1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step,
                              loss.item()))
                pass

        pass

    pass

train(num_epochs, cnn_direct, loaders)


def test():
    # Test the model
    cnn_direct.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn_direct(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass

        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)

    pass

test()


print("Now with hyper network")

cnn_hyper = CNN()
print(cnn_hyper)

hypernet = HyperNet(cnn_hyper, input_dim=1, hidden_sizes=[16, 64, 128])
print(hypernet)
print(cnn_hyper)
print("Number of model parameters: ",
      sum(p.numel() for p in cnn_hyper.parameters()))
print("Number of hyper_model parameters: ",
      sum(p.numel() for p in hypernet.parameters()))

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(hypernet.parameters(), lr = 0.01)

num_epochs = 15


def train(num_epochs, hypernet, loaders):
    hypernet.train()

    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y

            z = torch.tensor([0.5])
            output = hypernet(z, b_x)[0]
            loss = loss_func(output, b_y)

            # clear gradients for this training step
            optimizer.zero_grad()

            # backpropagation, compute gradients
            loss.backward()  # apply gradients
            optimizer.step()

            if (i + 1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step,
                              loss.item()))
                pass

        pass

    pass

train(num_epochs, hypernet, loaders)


def test():
    # Test the model
    hypernet.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            z = torch.tensor([0.5])
            test_output, last_layer = hypernet(z, images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass

        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)

    pass

test()


print("Untrained model: ")
cnn_notrain = CNN()

def test():
    # Test the model
    cnn_notrain.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:

            test_output, last_layer = cnn_notrain(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            pass

        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)

    pass

test()

print("Done")
