from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import pandas as pd
import ssl
import os
from torch import nn

ssl._create_default_https_context = ssl._create_unverified_context

device = 'cuda:0'  # 换成对应的GPU
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
model = RobertaModel.from_pretrained('roberta-large')
model.eval()
model.to(device)

total = pd.read_csv('dataset/train_data.csv')
total1 = pd.read_csv('dataset/train_pre.csv')
texts = total['text']
lenth = len(texts)
h = 0
maxlen = 0
num = 5866  # 原训练集总数
filename = "text_Roberta"
mask = False  # 设置是否使用mask，即是否取平均
if mask:
    try:
        os.makedirs("./mask" + filename + "ValData")
        os.makedirs("./mask" + filename + "TrainData")
    except FileExistsError:
        print("文件夹已存在")
        pass
else:
    try:
        os.makedirs("./" + filename + "ValData")
        os.makedirs("./" + filename + "TrainData")
    except FileExistsError:
        print("文件夹已存在")
        pass


def getFeature(text):
    text = tokenizer.tokenize(text)
    text_id = tokenizer.convert_tokens_to_ids(text)
    text_id = torch.tensor(text_id, dtype=torch.long)
    text_id = text_id.unsqueeze(dim=0)
    text_id = text_id.to(device)
    output = model(text_id)[0]  # (1,21,1024)
    # print(output)
    text_embedding = model(text_id)[0][12]  # 取第1层，也可以取别的层。
    text_embedding = text_embedding.detach().squeeze().t().cpu().numpy()  # 切断反向传播。# torch.Size([1, 8, 768])
    text_embedding = text_embedding.reshape(-1, 1024)  # 后面这个768需要告诉下一层
    if not mask:
        text_embedding = np.mean(text_embedding, 0)
    return text_embedding


for i in range(lenth):
    text = texts[i]
    with torch.no_grad():
        nowName = "./"
        if mask:
            nowName += "mask"
        if i % 5 == 0:
            nowName += filename + "ValData/{}.npy".format(i)
        else:
            nowName += filename + "TrainData/{}.npy".format(i)
        # if os.path.exists(nowName):
        #     continue
        feature = getFeature(text)
        if feature.shape[0] > maxlen:
            maxlen = feature.shape[0]
        np.save(nowName, feature, allow_pickle=True)

texts = total1['text']
lenth = len(texts)
filename = "text"
for i in range(lenth):
    text = texts[i]
    with torch.no_grad():
        nowName = "./"
        if mask:
            nowName += "mask"
        nowName += filename + "TrainData/{}.npy".format(i + num)
        if os.path.exists(nowName):
            continue
        feature = getFeature(text)
        if feature.shape[0] > maxlen:
            maxlen = feature.shape[0]
        np.save(nowName, feature, allow_pickle=True)
print(maxlen)