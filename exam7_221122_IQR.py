import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/train.csv')


# print(df.shape)
# print(df.isna().sum())
# print(df.head())


q1 = df['Fare'].quantile(.25)
q3 = df['Fare'].quantile(.75)

IQR = q3-q1

print(q1-IQR*1.5, q3+IQR*1.5)

outdata = df[(df['Fare'] < q1-IQR*1.5)] | df[(df['Fare'] > q3+IQR*1.5)]

# outdata = df[(df['Fare'] < q1-IQR*1.5)]
# outdata1 = df['Fare'] < q1-IQR*1.5
# print(outdata,outdata1)

print(sum(outdata['Sex']=='female'))



