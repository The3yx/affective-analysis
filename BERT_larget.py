import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import ssl
import os
from torch import nn
#ssl._create_default_https_context = ssl._create_unverified_context

def TODO():
    return

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

class MyBertModel():
    '''
    example:
    <<< model = BertModel()
    <<< model()
    '''
    def __init__(self, myPath, bert):
        self.myPath = myPath
        self.bert = bert

    def __call__(self, filename, tokenizer, pathVal, mask=False):
        self.forward(filename, tokenizer, pathVal, mask)


    def forward(self, filename, tokenizer, pathVal, mask=False):
        num=0
        maxlen = 0
        for i in self.myPath:#执行每一个测试集
            num, maxlen=self._getVec(i, False, num, maxlen, filename, tokenizer, mask)

        #执行训练集
        num, maxlen=self._getVec(pathVal, True, 0, maxlen, filename, tokenizer, mask)
        return maxlen
    

    def _getFeature(self, text, tokenizer, mask = False, ceng = False):#根据Bert模型的特征，提取文字的embeding
        text = tokenizer.tokenize(text)
        text_id = tokenizer.convert_tokens_to_ids(text)
        text_id = torch.tensor(text_id, dtype=torch.long)
        text_id = text_id.unsqueeze(dim=0)
        text_id=text_id.to('cpu')
        output = self.bert(text_id)[0]
        # print(output)
        if ceng:
            text_embedding = self.bert(text_id)[0] # 取第1层，也可以取别的层。
            text_embedding=torch.stack(text_embedding,dim=2)
            text_embedding = text_embedding.detach().squeeze().cpu().numpy()  # 切断反向传播。# torch.Size([1, 8, 768])
            text_embedding=text_embedding.reshape(-1,1024,24) #后面这个768需要告诉下一层
        else:
            text_embedding = self.bert(text_id)[0][12]  # 取第1层，也可以取别的层。
            text_embedding = text_embedding.detach().squeeze().t().cpu().numpy()  # 切断反向传播。# torch.Size([1, 8, 768])
            text_embedding=text_embedding.reshape(-1,1024) #后面这个768需要告诉下一层
        if not mask:
            text_embedding = np.mean(text_embedding, 0)
        return text_embedding


    def _getVec(self, path, flag, num, maxlen, filename, tokenizer, mask=False):
        print('getVec', type(tokenizer))
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
                feature=self._getFeature(text = text, tokenizer = tokenizer)#获取特征
                if feature.shape[0]>tempLen:
                    tempLen=feature.shape[0]
                np.save(nowName, feature, allow_pickle=True)
        if tempLen>maxlen:#本次训练中有超过最大特征长度的内容
            maxlen=tempLen
        return lenth+num,maxlen



