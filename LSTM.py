import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import ssl
import os
import shutil 

ssl._create_default_https_context = ssl._create_unverified_context

device = 'cuda'
path="/data6/hupeng/data/vctkAll2Wav2cdNewvec2"
batch_size=512
input_dim = 512
hidden_dim = 30
num_layers = 1
output_dim = 2
lr=3e-3
epoch=100
logPath="./log"

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

trainData=MyDataset(os.path.join(path,"train"))
testData=MyDataset(os.path.join(path,"val"))

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



newModel =LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
newModel=newModel.to(device)

theLoss=nn.CrossEntropyLoss()
theLoss=theLoss.to(device)

optim =torch.optim.Adam(newModel.parameters(), lr=lr)

# if os.path.exists(logPath):
#     for i in range(logPath):
#         os.remove(os.path.join(logPath,i))
write =SummaryWriter(logPath)
for i in range(epoch):
    if i%10==0:
        lr = lr*0.7
        optim = torch.optim.Adam(newModel.parameters(), lr=lr)
    newLoss=0
    acc=0
    newModel.train()
    for data in trainDataLoader:
        wav,flag=data
        wav=wav.reshape(-1,1,512)
        flag=flag.reshape(-1)
        wav=wav.to(device)
        flag=flag.to(device)
        output=newModel(wav)
        output=output.reshape(batch_size,-1)
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
            wav = wav.reshape(-1, 1, 512)
            flag=torch.tensor(flag)
            wav=wav.to(device)
            flag=flag.to(device)
            output=newModel(wav)
            output = output.reshape(batch_size, -1)
            loss=theLoss(output,flag)
            newLoss+=loss.detach().squeeze().t().cpu().numpy()
            acc+=(output.argmax(1) == flag).sum().detach().squeeze().t().cpu().numpy()
    print("sum:"+str(i) + "   loss:" + str(newLoss) + "   acc:" + str(acc / testLen))
    write.add_scalar("loss",newLoss,i)
    write.add_scalar("acc",acc/testLen,i)
write.close()
