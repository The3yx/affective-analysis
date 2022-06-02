import nlpaug.augmenter.word as naw
import numpy as np
import pandas as pd
import sys
import os
import nltk
import gensim
import re
from nltk.corpus import stopwords
import nlpaug.augmenter.word as naw
from tqdm import tqdm
ACTION = 'substitute'
TOP_K = 10 
AUG_P = 0.40 
aug_bert = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', 
    action=ACTION, 
    top_k=TOP_K,
    aug_p=AUG_P
    )
sentiment = pd.read_csv("./twitter-sentiment-analysis-self-driving-cars/train.csv")

sentiment['text'] = sentiment['text'].replace(r'((www\.[^\s]+)|(https?://[^\s]+))', '', regex=True)
sentiment['text'] = sentiment['text'].replace('@[^\s]+', '', regex=True)
sentiment['text'] = sentiment['text'].replace('@ [^\s]+', '', regex=True)
q = sentiment['sentiment'].value_counts(sort=False)
max_q = q.max()
max_q = max_q*2
for i in range(1,6):
    l = sentiment.loc[sentiment['sentiment']==i]
    mod = (max_q-q[i])%q[i]
    num = int((max_q-q[i])/q[i])
    print(l)
    for idx,text in enumerate(l['text']):
        print(str(i)+" "+str(idx))
        if(idx<mod):
            augmented_text = aug_bert.augment(text, n=num+1)
        else:
            augmented_text = aug_bert.augment(text, n=num)
        if type(augmented_text) == str:
            print(augmented_text)
            sentiment = sentiment.append({'sentiment':i,'text':augmented_text},ignore_index=True)
        else:
            for text in augmented_text:
                print(text)
                sentiment = sentiment.append({'sentiment':i,'text':text},ignore_index=True)
q = sentiment['sentiment'].value_counts(sort=False)
print(q)
sentiment.to_csv('./dataset/train_data.csv', index=False)