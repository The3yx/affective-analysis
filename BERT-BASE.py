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

tokenizer = BertTokenizer.from_pretrained(r'C:\Users\86131\PycharmProjects\torch1\bert-base-chinese', do_lower_case=True)
bert = BertModel.from_pretrained(r'C:\Users\86131\PycharmProjects\torch1\bert-base-chinese')
bert.eval()

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


device = 'cuda' #读取数据
total = pd.read_csv(r'C:\Users\86131\Desktop\twitter-sentiment-analysis-self-driving-cars\train.csv')
sentiment = total['sentiment']
texts = total['text']
lenth = len(texts)
h=0
maxlen=0
filename="text"
mask=False #设置是否使用mask，即是否取平均
if mask:
    try:
        os.makedirs("./mask" + filename + "ValData")
        os.makedirs("./mask" + filename + "TrainData")
    except FileExistsError:
        pass
else:
    try:
        os.makedirs("./" + filename + "ValData")
        os.makedirs("./" + filename + "TrainData")
    except FileExistsError:
        pass
for i in range(lenth):
    # target = torch.tensor(int(sentiment[i])).reshape(1)
    text=texts[i]
    with torch.no_grad():
        text = tokenizer.tokenize(text)
        text_id = tokenizer.convert_tokens_to_ids(text)
        text_id = torch.tensor(text_id, dtype=torch.long)
        text_id = text_id.unsqueeze(dim=0)
        output = bert(text_id)[0]
        text_embedding = bert(text_id)[0][0]  # 取第1层，也可以取别的层。
        text_embedding = text_embedding.detach().squeeze().t().cpu().numpy()  # 切断反向传播。# torch.Size([1, 8, 768])
        text_embedding=text_embedding.reshape(-1,768)
        feature =text_embedding
        if not mask:
            feature = np.mean(text_embedding, 0)
        if feature.shape[0]>maxlen:
            maxlen=feature.shape[0]
        if mask:
            if h==4:
                h=0
                np.save("./mask"+filename+"ValData/{}.npy".format(i), feature, allow_pickle=True)
            else:
                h+=1
                np.save("./mask"+filename+"TrainData/{}.npy".format(i), feature, allow_pickle=True)
        else:
            if h==4:
                h=0
                np.save("./"+filename+"ValData/{}.npy".format(i), feature, allow_pickle=True)
            else:
                h+=1
                np.save("./"+filename+"TrainData/{}.npy".format(i), feature, allow_pickle=True)
print(maxlen)
        # np.save("./textValData/f{}.npy".format(i),target, allow_pickle=True)
        # flag.append(target.numpy())
        # z1.append(z_feature)
