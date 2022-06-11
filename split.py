import numpy as np
import pandas as pd
import random
random.seed(1)
sentiment = pd.read_csv(r"twitter-sentiment-analysis-self-driving-cars\train.csv")
eval_set = sentiment.sample(frac=0.2)
eval_set.to_csv('./dataset/ori_eval_split.csv', index=False)
sentiment = sentiment.append(eval_set)
sentiment=sentiment.drop_duplicates(subset=['sentiment','text'],keep=False)
sentiment.to_csv('./dataset/ori_train_split.csv', index=False)