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
from sklearn.metrics import classification_report






# Read label ground truth
label_dataset_path = '/home/yoson/SparseInst/official/SparseInst/table-tennis/ball_data/pose_label/label/test'
image_dataset_path = '/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R_gray/'


total_frame = len(os.listdir(image_dataset_path))
gt = np.zeros(total_frame)



train_data = []
train_label = []
batch=12
window_size = 12

for _, walk_item in enumerate(os.walk(label_dataset_path)):

    root, dirs, files = walk_item

    if not dirs and files:

        label_path = None

        for file in files:

            if "_label_less0" in file:
                label_path = os.path.join(root, file)

        if label_path:
            ## 讀取動作區間標記
            label_df = pd.read_csv(f'{label_path}')
            label_df['label'] = label_df['label'].astype(int)

            n = 4
            k = batch//2
            ## 動作區間篩選
            for _, row in label_df.iterrows():

                if int(row['label']) in [5]:         # 右正手發球
                    label = [1]
                elif int(row['label']) in [6]:       # 右反手發球
                    label = [2]
                elif int(row['label']) in [7]:       # 右正手回球
                    label = [3]
                elif int(row['label']) in [8]:       # 右反手回球
                    label = [4]
                else:
                    label = [0]

                for i in range(row['start'], row['end']):
                    gt[i] = label[0]

# print(gt.reshape(gt.shape[0]))
gt_label = gt[6:-6]
# print(max(gt_label))
# print(gt_label.shape)
# exit()


# 512X512
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
        self.fc1 = nn.Linear(256 * 128 * 128, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

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
        return x

# initialize the Conv3DNet model
print("[INFO] initializing the Conv3DNet model...")
num_classes = 5
model = Conv2DNet(num_classes)
model.load_state_dict(torch.load('/home/yoson/SparseInst/official/SparseInst/table-tennis/pose_data/model/test_pose_model_less0.pth'))
print(model.eval())

# define training hyperparameters
INIT_LR = 1e-4
BATCH_SIZE = 4
EPOCHS = 50
num_classes = 5
model = Conv2DNet(num_classes)



# Datasets Preprocessing
dataset_path = '/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R_gray/'

# train_data = []
# train_label = []

preds = []
window_size = 12
for i in range(int((window_size/2)) + 0, int(len(os.listdir(dataset_path)) - (window_size/2))):
    img_list = []
    for j in range (int(i-(window_size/2))+1, int(i+(window_size/2))+1):
        print(j)
        # img_list.append(cv2.imread(f'/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R_gray/merge_R_frame_{j}.png', cv2.IMREAD_GRAYSCALE))
        img = cv2.imread(f'/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R_gray/merge_R_frame_{j}.png', cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(f'/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R_rgb/merge_R_frame_{j}.png')

        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)    # (1080, 960) (256, 256)
        img_list.append(img)


    print("----------------------------------------------------------------")
    img_list_np = np.asarray(img_list)
    with torch.no_grad():
        # set the model in evaluation mode
        model = model.cuda()
        model.eval()
        # initialize a list to store our predictions

        # loop over the test set

            # send the input to the device
        img_list_torch = torch.from_numpy(img_list_np.reshape(-1, 12, 512, 512)).float().cuda(non_blocking=True)
        # img_list_np = torch.FloatTensor(img_list_np)
        # make the predictions and add them to the list
        pred = model(img_list_torch)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
        # print(preds)

# generate a classification report
print(len(gt_label.reshape(gt_label.shape[0])))
# print(gt_label)
print(np.array(preds).shape)
# print(preds)
print(classification_report(gt_label, preds, labels=[1, 2, 3, 4, 0]))


import seaborn as sns
from sklearn.metrics import confusion_matrix

labels = ['1', '2', '3', '4', '0']
cm = confusion_matrix(gt_label, preds)
f = sns.heatmap(cm, annot=True, fmt='d', cmap="BuPu")
f.set_xticklabels(labels)
f.set_yticklabels(labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

exit()


    # train_data.append(img_list)
    # train_label.append(label)



# train_data.append("/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R/", )
# train_label.append(label)


# print("--------------------------------------------------------------------")
# np_train_data = np.asarray(train_data).reshape(-1, window_size, 256, 256, 3)
# np_train_data = np.asarray(train_data)
# data = np.array(data).astype(np.float32()) / 255.0
# print(np_train_data.shape)
# np_train_label = np.asarray(train_label).reshape(-1, 1)
# print(np_train_label.shape)
# print(np_train_label)

# # Save the data as numpy arrays
# np.save('/home/yoson/SparseInst/official/SparseInst/table-tennis/pose_data/pose_train_data.npy', np_train_data)
# np.save('/home/yoson/SparseInst/official/SparseInst/table-tennis/pose_data/pose_train_label.npy', np_train_label)