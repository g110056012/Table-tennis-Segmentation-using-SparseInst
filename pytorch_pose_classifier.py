from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import glob
import os
import cv2

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.utils.data import random_split
# set the numpy seed for better reproducibility
import numpy as np
np.random.seed(42)
# import the necessary packages
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2
import time


# Load the saved numpy arrays
np_train_data = np.load('/home/yoson/SparseInst/official/SparseInst/table-tennis/pose_data/pose_train_data.npy')
np_train_label = np.load('/home/yoson/SparseInst/official/SparseInst/table-tennis/pose_data/pose_train_label.npy')
print(np_train_data.shape)
print(np_train_label.shape)

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 30



np_train_data_reshape = np_train_data
train_data_list = []
# test_data_list = []
for i in range(len(np_train_data_reshape)):
#     print(np_train_data_reshape[i].shape())
    train_data_list.append([np_train_data_reshape[i], np_train_label[i]])

dataset = train_data_list
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
i1, l1 = next(iter(train_dataset))
# print(np.shape((train_dataset)), np.shape(val_dataset))
# np.shape(train_dataset[0][1])
print(np.shape(train_dataset[0][0]))
# print(train_dataset[0][0])


# Split test_dataset
# train_size2 = int(0.9 * len(train_dataset))
# test_size = len(train_dataset) - train_size2
# train_dataset, test_dataset = random_split(train_dataset, [train_size2, test_size])
# i1, l1 = next(iter(train_dataset))
# print(len(train_dataset), len(val_dataset), len(test_dataset))
# print(i1.shape)


# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE


class Conv2DNet(nn.Module):
    def __init__(self, num_classes):
        super(Conv2DNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 64 * 64, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x



# initialize the Conv3DNet model
print("[INFO] initializing the Conv3DNet model...")
num_classes = 5
# model = Conv3DNet(num_classes)
model = Conv2DNet(num_classes)
print(model)


# initialize our optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr = INIT_LR)
criterion = nn.CrossEntropyLoss()
# initialize a dictionary to store training history
H = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
model = model.cuda()
# model = model.to(device)
criterion = criterion.to(device)




# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

train_loss_list = []
# loop over our epochs
for e in range(0, EPOCHS):
    print('------------------------------------------------------------------------------------------------')
    print(f'Epoch {e + 1}/{EPOCHS}:', end = ' ')

    # set the model in training mode
    model = model.cuda()
    model.train()

    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0

    # initialize the number of correct predictions in the training and validation step
    trainCorrect = 0
    valCorrect = 0

    # loop over the training set
    for (x, y) in trainDataLoader:
        # send the input to the device
#         (x, y) = (x.to(device), y.to(device))
        x = x.float().cuda(non_blocking = True)
        y = y.view(-1).cuda(non_blocking = True)

        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = criterion(pred, y)
        # print(pred)
        # print('--------------------------')
        # print(labels)

        # zero out the gradients, perform the backpropagation step, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add the loss to the total training loss so far and calculate the number of correct predictions
        totalTrainLoss += loss
        trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()


    # Printing loss for each epoch
    train_loss_list.append(totalTrainLoss/len(trainDataLoader))
    print(f"Training loss = {train_loss_list[-1]}")



#     # switch off autograd for evaluation
#     with torch.no_grad():
#         # set the model in evaluation mode
#         model = model.cuda()
#         model.eval()
#         # loop over the validation set
#         for (x, y) in valDataLoader:
#             # send the input to the device
# #             (x, y) = (x.to(device), y.to(device))
#             x = x.float().cuda(non_blocking = True)
#             y = y.view(-1).cuda(non_blocking = True)

#             # make the predictions and calculate the validation loss
#             pred = model(x)
#             totalValLoss += criterion(pred, y)

#             # calculate the number of correct predictions
#             valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()



#     # calculate the average training and validation loss
#     avgTrainLoss = totalTrainLoss / trainSteps
#     avgValLoss = totalValLoss / valSteps
#     # calculate the training and validation accuracy
#     trainCorrect = trainCorrect / len(trainDataLoader.dataset)
#     valCorrect = valCorrect / len(valDataLoader.dataset)
#     # update our training history
#     H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
#     H["train_acc"].append(trainCorrect)
#     H["val_loss"].append(avgValLoss.cpu().detach().numpy())
#     H["val_acc"].append(valCorrect)

#     # print the model training and validation information
#     print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
#     print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
#         avgTrainLoss, trainCorrect))
#     print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
#         avgValLoss, valCorrect))


# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))


torch.save(model.state_dict(), '/home/yoson/SparseInst/official/SparseInst/table-tennis/result/model.pth')


# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # initialize a list to store our predictions
    preds = []
    # loop over the test set
    for (x, y) in trainDataLoader:
        # send the input to the device
#         x = x.to(device)
        x = x.float().cuda(non_blocking=True)
        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

# generate a classification report
print(classification_report([y[0] for x,y in train_dataset], preds, labels=[1, 2, 3, 4, 0]))








































# class Conv2DNet(nn.Module):
#     def __init__(self, num_classes):
#         super(Conv2DNet, self).__init__()
#         self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
#         self.relu1 = nn.ReLU()
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU()
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)
#         self.dropout = nn.Dropout2d(p=0.2)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.relu3 = nn.ReLU()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(256 * 64 * 64, 64)
#         self.relu4 = nn.ReLU()
#         self.fc2 = nn.Linear(64, num_classes)
#         # self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)
#         x = self.dropout(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu4(x)
#         x = self.fc2(x)
#         # x = self.softmax(x)
#         return x

# num_classes = 5
# model = Conv2DNet(num_classes)


# np_train_data_reshape = np_train_data
# # for i in np_train_data_reshape:
# #     # print(i.shape)

# # define training hyperparameters
# INIT_LR = 1e-4
# BATCH_SIZE = 2
# EPOCHS = 100
# # define the train and val splits
# TRAIN_SPLIT = 0.75
# VAL_SPLIT = 1 - TRAIN_SPLIT


# train_data_list = []
# for i in range(len(np_train_data_reshape)):
# #     print(np_train_data_reshape[i].shape())
#     train_data_list.append([np_train_data_reshape[i], np_train_label[i]])

# dataset = train_data_list
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
# i1, l1 = next(iter(train_dataset))
# # print(len(train_dataset))
# # print(i1.shape)


# # initialize the train, validation, and test data loaders
# trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
# valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# # testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)
# # calculate steps per epoch for training and validation set
# trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
# valSteps = len(valDataLoader.dataset) // BATCH_SIZE


# # initialize the Conv3DNet model
# print("[INFO] initializing the Conv3DNet model...")
# num_classes = 5
# # model = Conv3DNet(num_classes)
# model = Conv2DNet(num_classes)
# print(model)
# # print(model)

# # initialize our optimizer and loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.CrossEntropyLoss()
# # initialize a dictionary to store training history
# H = {
#     "train_loss": [],
# 	"train_acc": [],
# 	"val_loss": [],
# 	"val_acc": []
# }
# # measure how long training is going to take
# print("[INFO] training the network...")
# startTime = time.time()


# # Check if CUDA is available
# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
# print(device)
# model = model.cuda()
# # model = model.to(device)
# criterion = criterion.to(device)


# #   Training process begins
# train_loss_list = []
# num_epochs = EPOCHS
# for epoch in range(num_epochs):
#     print('------------------------------------------------------------------------------------------------')
#     print(f'Epoch {epoch + 1}/{num_epochs}:', end = ' ')
#     train_loss = 0

#     #Iterating over the training dataset in batches
#     model = model.cuda()
#     model.train()
#     for (images, labels) in trainDataLoader:
#         #Extracting images and target labels for the batch being iterated
#         images = images.float().cuda(non_blocking = True)
#         labels = labels.view(-1).cuda(non_blocking = True)

#         #Calculating the model output and the cross entropy loss
#         pred = model(images)
#         loss = criterion(pred, labels)
#         # print(pred)
#         # print('--------------------------')
#         # print(labels)

#         #Updating weights according to calculated loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     #Printing loss for each epoch
#     train_loss_list.append(train_loss/len(trainDataLoader))
#     print(f"Training loss = {train_loss_list[-1]}")

# #Plotting loss for all epochs
# plt.plot(range(1, num_epochs + 1), train_loss_list)
# plt.xlabel("Number of epochs")
# plt.ylabel("Training loss")















# #   Version 1
# # loop over our epochs
# for e in range(0, EPOCHS):
#     model = model.cuda()
#     # set the model in training mode
#     model.train()
#     # initialize the total training and validation loss
#     totalTrainLoss = 0
#     totalValLoss = 0
#     # initialize the number of correct predictions in the training
#     # and validation step
#     trainCorrect = 0
#     valCorrect = 0
#     print('***********************************')
#     # loop over the training set
#     for (x, y) in trainDataLoader:
#         # send the input to the device
# #         (x, y) = (x.to(device), y.to(device))
# #         (x, y) = (x.to(device).type(torch.FloatTensor), y.to(device))
#         x = x.float().cuda(non_blocking=True)
#         y = y.view(-1).cuda(non_blocking=True)

#         # perform a forward pass and calculate the training loss
#         pred = model(x)
#         loss = criterion(pred, y)
#         print(pred)
#         print('--------------------------')
#         print(y)

#         # zero out the gradients, perform the backpropagation step,
#         # and update the weights
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # add the loss to the total training loss so far and
#         # calculate the number of correct predictions
#         totalTrainLoss += loss
#         trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

