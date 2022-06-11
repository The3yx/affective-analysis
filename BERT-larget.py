import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import soundfile as sf
import torch
import librosa
import numpy as np
from fairseq.models.wav2vec import Wav2VecModel #fairseq.__version__ = 0.10.1
import pandas as pd
import ssl
import os
from torch import nn
ssl._create_default_https_context = ssl._create_unverified_context

#获取模型等操作
device = 'cuda:5' #换成对应的GPU
tokenizer = BertTokenizer.from_pretrained('/data6/hupeng/code/bert-large-uncased', do_lower_case=True)
bert = BertModel.from_pretrained('/data6/hupeng/code/bert-large-uncased')
bert.eval()
bert.to(device)
#所有训练集路径，测试集路径
myPath=['/data6/hupeng/data/twitter-sentiment-analysis-self-driving-cars/dataset/train_split.csv']
pathVal='/data6/hupeng/data/twitter-sentiment-analysis-self-driving-cars/dataset/eval_split.csv'
filename="text" #部分文件名
mask=False #设置是否使用mask，即是否取平均
ceng=False #保留所有层

def removeFile(file):#则删除文件夹内的文件
    for i in os.listdir(file):
        os.remove(os.path.join(file,i))

def getFeature(text):#根据Bert模型的特征，提取文字的embeding
    text = tokenizer.tokenize(text)
    text_id = tokenizer.convert_tokens_to_ids(text)
    text_id = torch.tensor(text_id, dtype=torch.long)
    text_id = text_id.unsqueeze(dim=0)
    text_id=text_id.to(device)
    output = bert(text_id)[0]
    # print(output)
    if ceng:
        text_embedding = bert(text_id)[0] # 取第1层，也可以取别的层。
        text_embedding=torch.stack(text_embedding,dim=2)
        text_embedding = text_embedding.detach().squeeze().cpu().numpy()  # 切断反向传播。# torch.Size([1, 8, 768])
        text_embedding=text_embedding.reshape(-1,1024,24) #后面这个768需要告诉下一层
    else:
        text_embedding = bert(text_id)[0][12]  # 取第1层，也可以取别的层。
        text_embedding = text_embedding.detach().squeeze().t().cpu().numpy()  # 切断反向传播。# torch.Size([1, 8, 768])
        text_embedding=text_embedding.reshape(-1,1024) #后面这个768需要告诉下一层
    if not mask:
        text_embedding = np.mean(text_embedding, 0)
    return text_embedding

def getvec(path,flag,num,maxlen):
    #参数分别为输入文件路径，是否为测试集，本文件第一个npy的位置在所有测试集中的位置，目前的最大长度
    total = pd.read_csv(path)
    texts = total['text']
    lenth = len(texts)
    #提取对应列参数
    tempLen=0
    #设置文件路径
    Name="./"
    if mask:
        Name+="mask"
    if flag:
        Name+=filename+"ValData"
    else:
        Name+=filename+"TrainData"

    for i in range(lenth):
        text=texts[i]
        with torch.no_grad():
            nowName=Name+"/{}.npy".format(i+num)
            if os.path.exists(nowName):#如果已经存在则不继续
                continue
            feature=getFeature(text)#获取特征
            if feature.shape[0]>tempLen:
                tempLen=feature.shape[0]
            np.save(nowName, feature, allow_pickle=True)
    if tempLen>maxlen:#本次训练中有超过最大特征长度的内容
        maxlen=tempLen
    return lenth+num,maxlen

#创建文件路径，如果已存在就删除里面的文件
flagPath=""
if mask:
    flagPath="mask"
try:
    os.makedirs("./"+flagPath + filename + "ValData")
    os.makedirs("./"+flagPath + filename + "TrainData")
except FileExistsError:
    removeFile("./"+flagPath + filename + "ValData")
    removeFile("./"+flagPath + filename + "TrainData")
    print("文件夹已存在,并已删除原文件")
    pass

maxlen=0
num=0
for i in myPath:#执行每一个测试集
    num,maxlen=getvec(i,False,num,maxlen)

#执行训练集
num,maxlen=getvec(pathVal,True,0,maxlen)

print(maxlen)

# for i in range(lenth):
#     text=texts[i]
#     with torch.no_grad():
#         nowName="./"
#         if mask:
#             nowName+="mask"
#         if i%5==0:
#             nowName+=filename+"ValData/{}.npy".format(i)
#         else:
#             nowName+=filename+"TrainData/{}.npy".format(i)
#         # if os.path.exists(nowName):
#         #     continue
#         feature=getFeature(text)
#         if feature.shape[0]>maxlen:
#             maxlen=feature.shape[0]
#         np.save(nowName, feature, allow_pickle=True)


# texts = total1['text']
# lenth = len(texts)
# for i in range(lenth):
#     text=texts[i]
#     with torch.no_grad():
#         nowName="./"
#         if mask:
#             nowName+="mask"
#         nowName+=filename+"TrainData/{}.npy".format(i+num)
#         if os.path.exists(nowName):
#             continue
#         feature=getFeature(text)
#         if feature.shape[0]>maxlen:
#             maxlen=feature.shape[0]
#         np.save(nowName, feature, allow_pickle=True)
# print(maxlen)



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
