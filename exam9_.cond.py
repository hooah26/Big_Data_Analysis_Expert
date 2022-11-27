import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic1.csv')



df =  df.sort_values('age', ascending=False).reset_index(drop=True)
# print(df)
df = df[:20]
# print(df)


df['f1'] = df['f1'].fillna(df['f1'].median())


cond = (df['f4']=='ISFJ') & (df['f5']>='20')

print(df[condi]['f1'].mean())
