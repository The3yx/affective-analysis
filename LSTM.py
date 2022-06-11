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
device = 'cpu'
input_dim = 1024


class LSTMmask(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, ceng =False):
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

    def forward(self, x,mask, ceng=False):
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
        self.drop = nn.Dropout(0.9)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out.to(device)
        out = self.drop(out)
        return self.decoder(out)

