import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/winequality-red.csv')
# print(df.head())

df_corr = df.corr()
df_corr = df_corr[:-1]
# print(df_corr)
max_corr =  abs(df.corr()['quality'][:-1]).max()
min_corr =  abs(df.corr()['quality'][:-1]).min()
# print(max_corr,min_corr)

