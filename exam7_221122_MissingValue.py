import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic1.csv')
# print(df.shape)
# print(df.isna().sum()/df.shape[0])

df= df.drop(['f3'], axis=1)
# print(df.shape)

# print(df['city'].unique())
s=df[df['city']=='서울']['f1'].median()
b=df[df['city']=='부산']['f1'].median()
d=df[df['city']=='대구']['f1'].median()
k=df[df['city']=='경기']['f1'].median()

# print(s,b,d,k)

df['f1'] = df['f1'].fillna(df['city'].map({'서울':s,'부산':b,'대구':d,'경기':k}))

# print(df['f1'].isna().sum())

print(df['f1'].mean())