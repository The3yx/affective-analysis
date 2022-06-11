import numpy as np
import pandas as pd
import random
random.seed(1)
sentiment = pd.read_csv(r"./dataset/clean_ori_data.csv")
eval_set = sentiment.sample(frac=0.2)
eval_set.to_csv('./dataset/clean_ori_data_train_split.csv', index=False)
sentiment = sentiment.append(eval_set)
sentiment=sentiment.drop_duplicates(subset=['sentiment','text'],keep=False)
sentiment.to_csv('./dataset/clean_ori_data_test_split.csv', index=False)