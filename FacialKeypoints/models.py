## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers
        # (such as dropout or batch normalization) to avoid overfitting

        # Convolution Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2)

        self.conv1_bn = nn.BatchNorm2d(self.conv1.out_channels)
        self.conv2_bn = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3_bn = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4_bn = nn.BatchNorm2d(self.conv4.out_channels)

        # Maxpool Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        # print("Flatten size: ", x.shape) >> 36864
        self.fc1 = nn.Linear(in_features=36864, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=136)

        self.fc1_bn = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2_bn = nn.BatchNorm1d(self.fc2.out_features)

        # Dropouts
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))

        x = x.view(x.size(0), -1)
        # print("Flatten size: ", x.shape)

        x = self.dropout(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.dropout(F.relu(self.fc2_bn(self.fc2(x))))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x

