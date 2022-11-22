from sklearn.preprocessing import power_transform
data =  [[11, 12], [23, 22], [34, 35]]
# print(power_transform(data))
# print(power_transform(data, method='box-cox'))

import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic1.csv')
print('전',df.shape)
df =  df[df['age']>=20]
print('후', df.shape)

print("결측치 처리 전: \n", df.isnull().sum())
print("최빈값: ",df['f1'].mode()[0])
df['f1'] = df['f1'].fillna(df['f1'].mode()[0])
print("결측치 처리 후: \n", df.isnull().sum())   

df['y'] = power_transform(df[['f1']]) 
df['y'].head()

df['y'] = power_transform(df[['f1']],standardize=False) 
df['y'].head()


df['b'] = power_transform(df[['f1']], method='box-cox')
df['b'].head()


df['b'] = power_transform(df[['f1']], method='box-cox', standardize=False)
df['b'].head()


from scipy import stats
x = stats.boxcox(df['f1'])
x

round(sum(np.abs(df['y'] - df['b'])),2)
