import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import ssl
import os
import shutil 


class MyDataset(Dataset):    #继承Datasets
    def __init__(self,path):    # 初始化一些用到的参数，一般不仅有self
        target=[]
        path1=[]
        for i in os.listdir(path):
            tdir=os.path.join(path,i)
            tLen=len(os.listdir(tdir))
            for j in range(tLen):
                target.append(int(i))
            for j in os.listdir(tdir):
                path1.append(os.path.join(tdir,j))
        self.target=target
        self.path=path1
    def __len__(self):    # 数据集的长度
        return len(self.target)
    def __getitem__(self, idx):    # 按照索引读取每个元素的具体内容
        target = self.target[idx]
        x = np.load(self.path[idx], allow_pickle=True)
        return x ,target

class MyDatasetWithMask(Dataset):    #继承Datasets
    def __init__(self,path,sentiment,sentiment1,maxLen):    # 初始化一些用到的参数，一般不仅有self
        target=[]
        path1=[]
        for i in os.listdir(path):
            path1.append(os.path.join(path,i))
            num=(int)(i.strip(".npy"))
            if num<sumNum:
                target.append(sentiment[num]-1)
            else:
                target.append(sentiment1[num-sumNum]-1)
        self.target=target
        self.path=path1
        self.maxLen=maxLen
    def __len__(self):    # 数据集的长度
        return len(self.target)
    def getMask(self,x):
        seq = x.shape[0]
        mask = np.array([True] * seq + [False] * (self.maxLen - seq)).reshape(-1, 1)
        mask = (mask + 0).astype('float32')
        return mask
    def __getitem__(self, idx):    # 按照索引读取每个元素的具体内容
        target = self.target[idx]
        x = np.load(self.path[idx], allow_pickle=True)
        mask=self.getMask(x)
        if x.shape[0]<self.maxLen:
            x=np.concatenate([x, np.zeros((self.maxLen-x.shape[0],(int)(x.shape[1])),'f')], axis=0)
        return x ,target,mask