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
device = 'cpu'
batch_size=32
total = pd.read_csv(r'C:\Users\86131\Desktop\twitter-sentiment-analysis-self-driving-cars\train.csv')
sentiment = total['sentiment']
trainData=MyDataset("textTrainData",sentiment)
testData=MyDataset("textValData",sentiment)

trainLen=len(trainData)
testLen=len(testData)
print(trainLen)
print(testLen)
trainDataLoader=DataLoader(trainData,batch_size=batch_size,shuffle=True,drop_last=True)
testDataLoader=DataLoader(testData,batch_size=batch_size,shuffle=True,drop_last=True)


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
        self.fc1=nn.Linear(hidden_dim,1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out.to(device)
        return self.decoder(out)

input_dim = 768
hidden_dim = 20
num_layers = 1
output_dim = 5

newModel =LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
newModel=newModel.to(device)

theLoss=nn.CrossEntropyLoss()
theLoss=theLoss.to(device)

lr=1e-3
optim =torch.optim.Adam(newModel.parameters(), lr=lr)

epoch=120
write =SummaryWriter("./log")
theStep=0
for i in range(epoch):
    if i%10==0:
        lr = lr*0.7
        optim = torch.optim.Adam(newModel.parameters(), lr=lr)
    newLoss=0
    acc=0
    theStep+=1
    newModel.train()
    for data in trainDataLoader:
        wav,flag=data
        flag = flag.reshape(-1)
        wav=wav.reshape(-1,1,768)
        wav=wav.to(device)
        flag=flag.to(device)
        output=newModel(wav)
        output = output.reshape(batch_size, -1)
        loss=theLoss(output,flag)
        optim.zero_grad()
        loss.backward()
        optim.step()
        del wav,flag,loss
    newModel.eval()
    for data in testDataLoader:
        with torch.no_grad():
            wav,flag=data
            flag=flag.reshape(-1)
            wav = wav.reshape(-1, 1, 768)
            flag=torch.tensor(flag)
            wav=wav.to(device)
            flag=flag.to(device)
            output=newModel(wav)
            output = output.reshape(batch_size, -1)
            loss=theLoss(output,flag)
            newLoss+=loss
            acc+=(output.argmax(1)==flag).sum()
    print(str(i) + "   loss:" + str(newLoss) + "   acc:" + str(acc / testLen))
    write.add_scalar("loss",newLoss,theStep)
    write.add_scalar("acc",acc/testLen,theStep)
write.close()
