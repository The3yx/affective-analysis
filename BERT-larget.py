import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import soundfile as sf
import torch
import librosa
import numpy as np
from fairseq.models.wav2vec import Wav2VecModel  # fairseq.__version__ = 0.10.1
import pandas as pd
import ssl
import os
from torch import nn

ssl._create_default_https_context = ssl._create_unverified_context

device = 'cuda:5'  # 换成对应的GPU
tokenizer = BertTokenizer.from_pretrained('/data6/hupeng/code/bert-large-uncased', do_lower_case=True)
bert = BertModel.from_pretrained('/data6/hupeng/code/bert-large-uncased')
bert.eval()
bert.to(device)

total = pd.read_csv('/data6/hupeng/data/twitter-sentiment-analysis-self-driving-cars/train.csv')
total1 = pd.read_csv('/data6/hupeng/data/twitter-sentiment-analysis-self-driving-cars/train_pre.csv')
texts = total['text']
lenth = len(texts)
h = 0
maxlen = 0
num = 942  # 原训练集总数
filename = "text"
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
    output = bert(text_id)[0]
    # print(output)
    text_embedding = bert(text_id)[0][12]  # 取第1层，也可以取别的层。
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
# np.save("./textValData/f{}.npy".format(i),target, allow_pickle=True)
# flag.append(target.numpy())
# z1.append(z_feature)


# text = '我爱北京天安门。'
# text = tokenizer.tokenize(text)
# print(text)  # ['我', '爱', '北', '京', '天', '安', '门', '。']
# text_id = tokenizer.convert_tokens_to_ids(text)
# print(text_id)  # [2769, 4263, 1266, 776, 1921, 2128, 7305, 511]
# text_id = torch.tensor(text_id, dtype=torch.long)
# text_id = text_id.unsqueeze(dim=0)
# print(text_id)  # tensor([[2769, 4263, 1266,  776, 1921, 2128, 7305,  511]])
# output = bert(text_id)[0]
# print(len(output))  # 12层
# text_embedding = bert(text_id)[0][0]  # 取第1层，也可以取别的层。
# text_embedding = text_embedding.detach()  # 切断反向传播。
# print(text_embedding.shape)  # torch.Size([1, 8, 768])
