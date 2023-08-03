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


# Datasets Preprocessing

dataset_path = '/home/yoson/SparseInst/official/SparseInst/table-tennis/ball_data/pose_label/label/test'

train_data = []
train_label = []
batch=12
window_size = 12
for _, walk_item in enumerate(os.walk(dataset_path)):

    root, dirs, files = walk_item

    if not dirs and files:

        label_path = None

        for file in files:

            if "_labelallframe.csv" in file:
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
                elif int(row['label']) in [0]:       # 死球
                    label = [0]
                else:
                    label = [0]


                for i in range((row['end'] - row['start'] - window_size) + 1):
                    print('group_num', (row['end'] - row['start'] - window_size) + 1)
                    img_list = []
                    begin = row['start'] + i
                    for j in range (begin, begin + 12):
                        # img_list.append(cv2.imread(f'/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R_gray/merge_R_frame_{j}.png', cv2.IMREAD_GRAYSCALE))

                        img = cv2.imread(f'/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R_gray/merge_R_frame_{j}.png', cv2.IMREAD_GRAYSCALE)
                        # img = cv2.imread(f'/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R_rgb/merge_R_frame_{j}.png')
                        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)    # (1080, 960) (256, 256)
                        img_list.append(img)


                        # cv2.namedWindow('person_mask_R', 0)
                        # cv2.imshow('person_mask_R', img)
                        # cv2.waitKey(0)
                        # exit()


                    print("start: ", begin)
                    print("end: ", begin + 12)
                    print("label: ", label)
                    print(row['label'])
                    print(len(img_list))
                    img_list_np = np.asarray(img_list)
                    print(img_list_np.shape)

                    train_data.append(img_list)
                    train_label.append(label)

            # train_data.append("/home/yoson/SparseInst/official/SparseInst/table-tennis/label_mask/merge_R/", )
            # train_label.append(label)


print("--------------------------------------------------------------------")
# np_train_data = np.asarray(train_data).reshape(-1, window_size, 256, 256, 3)
np_train_data = np.asarray(train_data)
# data = np.array(data).astype(np.float32()) / 255.0
print(np_train_data.shape)
np_train_label = np.asarray(train_label).reshape(-1, 1)
print(np_train_label.shape)
print(np_train_label)

# # Save the data as numpy arrays
np.save('/home/yoson/SparseInst/official/SparseInst/table-tennis/pose_data/256_pose_train_data_allframe.npy', np_train_data)
np.save('/home/yoson/SparseInst/official/SparseInst/table-tennis/pose_data/256_pose_train_label_allframe.npy', np_train_label)


