import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic1.csv')
# print(df)
# print(df.isna().sum())
df =  df[~df['f1'].isna()]
# print(df.isna().sum())
df2 = df.groupby(['city','f2']).sum()
# print(df2)

print(df2.iloc[0]['f1'])

