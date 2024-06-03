from __future__ import annotations, print_function, division
import os
from PIL import Image
import numpy as np
import random
import copy
import time
import torch

import json
class MyDataset():
    def __init__(self,root_dir,annotations_file,transform=None):

        self.root_dir = root_dir
        self.annotations = annotations_file
        self.transform = transform

        # if not os.path.isfile(self.annotations_file):
        #     print(self.annotations + "does not exist")
        self.json_file=os.listdir(root_dir +annotations_file)
        # print(self.json_file)
        # self.file_info = pd.read_csv(root_dir +annotations_file,index_col=0)
        self.size = len(self.json_file)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self,idx):
        root_dir=self.root_dir
        annotations_file=self.annotations
        
        # print(os.path.join(root_dir +annotations_file+self.json_file[idx]))
        with open(os.path.join(root_dir +annotations_file+self.json_file[idx])) as f:
            data=json.load(f)
            # print(data)
        img_path=data['img_path']
        # print(img_path)
        label_eyes = int(data['label_eyes'])
    
        label_glass=int(data['label_glass'])
        img = Image.open(root_dir+'img/'+img_path.split('/')[-1]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # print('label_eyes',type(label_eyes))
        return img, (label_glass,label_eyes)
# train_path='./dataset/train/'
# annotations_file='label/'
# A=MyDataset(train_path,annotations_file)
# for i in range(5000):
#     img,target=A.__getitem__(i)
#     print(i)
#     print(img)
#     # print('ieye',target[:,0])
#     # print('glass',target[:,1])
#     print(target)
