import numpy as np
import pandas as pd
import sys
import os
import nltk
import gensim
import re
from nltk.corpus import stopwords
sentiment = pd.read_csv("./twitter-sentiment-analysis-self-driving-cars/train.csv")
text_col = sentiment['text'].str.lower()
text_col = text_col.replace(r'((www\.[^\s]+)|(https?://[^\s]+))', '', regex=True)
text_col = text_col.replace('@[^\s]+', '', regex=True)
text_col = text_col.replace('@ [^\s]+', '', regex=True)
text_col = text_col.replace(r'#([^\s]+)', r'\1', regex=True)
text_col = text_col.replace('[\s]+', ' ', regex=True)
text_col = text_col.replace(r'\w*\d\w*', '', regex=True)
text_col = text_col.replace(r'ì|¢|‰|â||ò|¢|ã|¢|å|ü|á|û|ï|ü|ª','',regex=True)
for word in contractions:
    print(word, contractions[word])
    text_col = text_col.replace(word, contractions[word],regex=True)
text_col = text_col.replace('([^\s\w-]|_|)+', "", regex=True)
text_col = text_col.replace('-', " ", regex=True)
stop_words = set(stopwords.words('english')) - {"nor", "not", "no"} 
stopwords_re = re.compile(r"(\s+)?\b({})\b(\s+)?".format("|".join(stop_words), re.IGNORECASE))
whitespace_re = re.compile(r"\s+")
text_col = text_col.replace(stopwords_re, " ").str.strip().str.replace(whitespace_re, " ")
ls = []
for num in range(len(text_col)):
    ls.append(dict())
    ls[num]['sentiment']=sentiment['sentiment'][num]
    ls[num]['text']=text_col[num]
ls=pd.DataFrame(ls)
ls.to_csv('train_pre.csv', index=False)
