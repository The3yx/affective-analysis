import torch
import numpy as np
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import ssl
import torch.nn.functional as F
import os

ssl._create_default_https_context = ssl._create_unverified_context
sumNum = 942
device = 'cuda:5'
input_dim = 1024
hidden_dim = 20
num_layers = 1
output_dim = 5
lr = 1e-3
batch_size = 128
trainPath = "masktextTrainData"
valPath = "masktextValData"
maxLen = 86  # 按照之前设置求的maxlen进行设置进行分类 77


class MyDataset(Dataset):  # 继承Datasets
    def __init__(self, path, sentiment, sentiment1, maxLen):  # 初始化一些用到的参数，一般不仅有self
        target = []
        path1 = []
        for i in os.listdir(path):
            path1.append(os.path.join(path, i))
            num = (int)(i.strip(".npy"))
            if num < sumNum:
                target.append(sentiment[num] - 1)
            else:
                target.append(sentiment1[num - sumNum] - 1)
        self.target = target
        self.path = path1
        self.maxLen = maxLen

    def __len__(self):  # 数据集的长度
        return len(self.target)

    def getMask(self, x):
        seq = x.shape[0]
        mask = np.array([True] * seq + [False] * (self.maxLen - seq)).reshape(-1, 1)
        mask = (mask + 0).astype('float32')
        return mask

    def __getitem__(self, idx):  # 按照索引读取每个元素的具体内容
        target = self.target[idx]
        x = np.load(self.path[idx], allow_pickle=True)
        mask = self.getMask(x)
        if x.shape[0] < self.maxLen:
            x = np.concatenate([x, np.zeros((self.maxLen - x.shape[0], (int)(x.shape[1])), 'f')], axis=0)
        return x, target, mask


total = pd.read_csv('/data6/hupeng/data/twitter-sentiment-analysis-self-driving-cars/train.csv')
total1 = pd.read_csv('/data6/hupeng/data/twitter-sentiment-analysis-self-driving-cars/train_pre.csv')
sentiment = total['sentiment']
sentiment1 = total1['sentiment']
trainData = MyDataset(trainPath, sentiment, sentiment1, maxLen)
testData = MyDataset(valPath, sentiment, sentiment1, maxLen)

trainLen = len(trainData)
testLen = len(testData)
print(trainLen)
print(testLen)
trainDataLoader = DataLoader(trainData, batch_size=batch_size)
testDataLoader = DataLoader(testData, batch_size=batch_size)


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.fc1 = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out.to(device)
        # x = out.permute(1, 0, 2)  # [T, B, Feat] ==> [B, T, Feat]
        hidden = self.fc1(out) * mask  # [B, T, Feat] => [B, T, 1]
        att = F.softmax(hidden, dim=2)  # [B, T, 1]
        attX = torch.matmul(out.transpose(1, 2), att)  # [B, Feat, 1]
        attX = attX.squeeze(dim=-1)  # [B, Feat]
        out = self.decoder(attX)  # [B, label]
        return out


newModel = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
newModel = newModel.to(device)

theLoss = nn.CrossEntropyLoss()
theLoss = theLoss.to(device)

optim = torch.optim.Adam(newModel.parameters(), lr=lr)

epoch = 100
write = SummaryWriter("../now/log")
theStep = 0
for i in range(epoch):
    newLoss = 0
    acc = 0
    theStep += 1
    newModel.train()
    for data in trainDataLoader:
        wav, flag, mask = data
        flag = flag.reshape(-1)
        wav = wav.to(device)
        flag = flag.to(device)
        mask = mask.to(device)
        output = newModel(wav, mask)
        loss = theLoss(output, flag)
        optim.zero_grad()
        loss.backward()
        optim.step()
        del wav, flag, mask, loss
    newModel.eval()
    for data in testDataLoader:
        wav, flag, mask = data
        flag = flag.reshape(-1)
        wav = torch.tensor(wav)
        flag = torch.tensor(flag)
        mask = mask.to(device)
        wav = wav.to(device)
        flag = flag.to(device)
        output = newModel(wav, mask)
        loss = theLoss(output, flag)
        newLoss += loss
        acc += (output.argmax(1) == flag).sum()
        del wav, flag, mask, loss
    print(str(i) + "   loss:" + str(newLoss) + "   acc:" + str(acc / testLen))
    write.add_scalar("loss", newLoss, theStep)
    write.add_scalar("acc", acc / testLen, theStep)
write.close()
torch.save(newModel, "../now/myModel1.pth")
