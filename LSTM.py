import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import ssl
import os
import pandas as pd
ssl._create_default_https_context = ssl._create_unverified_context
device = 'cuda:5'
input_dim = 1024
hidden_dim = 30
num_layers = 1
output_dim = 5
batch_size=64
lr=5e-3
maxlen=10
trainPath="textTrainData"
valPath="textValData"
myPath=['/data6/hupeng/data/twitter-sentiment-analysis-self-driving-cars/dataset/train_split.csv']
valCsvPath='/data6/hupeng/data/twitter-sentiment-analysis-self-driving-cars/dataset/eval_split.csv'
logPath="./bertLog"
ceng=False
mask=False

class MyDatasetmask(Dataset):    #继承Datasets
    def __init__(self,path,sentiment,maxLen):    # 初始化一些用到的参数，一般不仅有self
        target=[]
        path1=[]
        for i in os.listdir(path):
            path1.append(os.path.join(path,i))
            num=(int)(i.strip(".npy"))
            target.append(sentiment[num]-1)
        self.target=target
        self.path=path1
        self.maxLen=maxLen
    def __len__(self):    # 数据集的长度
        return len(self.target)
    def getMask(self,x):
        seq = x.shape[0]
        if ceng:
            mask = np.array([True] * seq + [False] * (self.maxLen - seq)).reshape(-1, 1)
        else:
            mask = np.array([True] * seq + [False] * (self.maxLen - seq)).reshape(-1, 1,1)
        mask = (mask + 0).astype('float32')
        return mask
    def __getitem__(self, idx):    # 按照索引读取每个元素的具体内容
        target = self.target[idx]
        x = np.load(self.path[idx], allow_pickle=True)
        mask=self.getMask(x)
        if x.shape[0]<self.maxLen:
            x=np.concatenate([x, np.zeros((self.maxLen-x.shape[0],(int)(x.shape[1])),'f')], axis=0)
        return x ,target,mask

class MyDataset(Dataset):    #继承Datasets
    def __init__(self,path,sentiment):    # 初始化一些用到的参数，一般不仅有self
        target=[]
        path1=[]
        for i in os.listdir(path):
            path1.append(os.path.join(path,i))
            num=(int)(i.strip(".npy"))
            target.append(sentiment[num]-1)
        self.target=target
        self.path=path1
    def __len__(self):    # 数据集的长度
        return len(self.target)
    def __getitem__(self, idx):    # 按照索引读取每个元素的具体内容
        target = self.target[idx]
        x = np.load(self.path[idx], allow_pickle=True)
        return x ,target

def getSentiment(path):
    total = pd.read_csv(path)
    sentiment = total['sentiment']
    return sentiment

trainSentiment=[]
for i in myPath:
    tempSentiment=getSentiment(i)
    if trainSentiment==[]:
        trainSentiment=tempSentiment
    else:
        trainSentiment+=tempSentiment
valSentiment=getSentiment(valCsvPath)
if mask:
    trainData=MyDatasetmask(trainPath,trainSentiment,maxlen)
    testData=MyDatasetmask(valPath,valSentiment,maxlen)
else:
    trainData=MyDataset(trainPath,trainSentiment)
    testData=MyDataset(valPath,valSentiment)

trainLen=len(trainData)
testLen=len(testData)
print(trainLen)
print(testLen)
trainDataLoader=DataLoader(trainData,batch_size=batch_size,shuffle=True,drop_last=True)
testDataLoader=DataLoader(testData,batch_size=batch_size,shuffle=True,drop_last=True)

class LSTMmask(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder=nn.Sequential(
            nn.Linear(hidden_dim,output_dim),
            nn.Sigmoid()
        )
        self.fc1 = nn.Linear(hidden_dim, 1)
        if ceng:
            self.w = torch.nn.Parameter(torch.FloatTensor(1,1,24), requires_grad=True)

    def forward(self, x,mask):
        if ceng:
            x=torch.sum(x*self.w,dim=2)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out.to(device)
        #x = out.permute(1, 0, 2)  # [T, B, Feat] ==> [B, T, Feat]
        hidden = self.fc1(out)* mask    # [B, T, Feat] => [B, T, 1]
        att =  F.softmax(hidden, dim=2)  # [B, T, 1]
        attX = torch.matmul(out.transpose(1, 2), att)  # [B, Feat, 1]
        attX = attX.squeeze(dim=-1)  # [B, Feat]
        out = self.decoder(attX)  # [B, label]
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder=nn.Sequential(
            nn.Linear(hidden_dim,output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out.to(device)
        return self.decoder(out)

if mask:
    newModel =LSTMmask(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
else:
    newModel =LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

newModel=newModel.to(device)

theLoss=nn.CrossEntropyLoss()
theLoss=theLoss.to(device)

optim =torch.optim.Adam(newModel.parameters(), lr=lr)

epoch=360
write =SummaryWriter(logPath)
theStep=0
for i in range(epoch):
    if i%30==0:
        lr = lr*0.7
        optim = torch.optim.Adam(newModel.parameters(), lr=lr)
    newLoss=0
    acc=0
    theStep+=1
    newModel.train()
    for data in trainDataLoader:
        wav,flag=data
        flag = flag.reshape(-1)
        wav=wav.reshape(-1,1,input_dim)
        wav=wav.to(device)
        flag=flag.to(device)
        output=newModel(wav)
        output = output.reshape(batch_size, -1)
        loss=theLoss(output,flag)
        optim.zero_grad()
        loss.backward()
        optim.step()
        acc+=(output.argmax(1)==flag).sum()
        newLoss+=loss
        del wav,flag,loss

    print("训练集："+str(i) + "   loss:" + str(newLoss) + "   acc:" + str(acc / trainLen))
    write.add_scalar("训练loss",newLoss,theStep)
    write.add_scalar("训练acc",acc/trainLen,theStep)
    acc=0
    newLoss=0
    newModel.eval()
    for data in testDataLoader:
        with torch.no_grad():
            wav,flag=data
            flag=flag.reshape(-1)
            wav = wav.reshape(-1, 1, input_dim)
            flag=torch.tensor(flag)
            wav=wav.to(device)
            flag=flag.to(device)
            output=newModel(wav)
            output = output.reshape(batch_size, -1)
            loss=theLoss(output,flag)
            newLoss+=loss
            acc+=(output.argmax(1)==flag).sum()
    print("测试集："+str(i) + "   loss:" + str(newLoss) + "   acc:" + str(acc / testLen))
    write.add_scalar("loss",newLoss,theStep)
    write.add_scalar("acc",acc/testLen,theStep)
write.close()
