import pandas as pd
import numpy as np


df =  pd.read_csv('/workspace/goorm/basic1.csv')
# print(df.head())
print(df.isna().sum())

df = df[(df['age']-np.floor(df['age']))!=0]
# print(df.head())

# df_ceil = np.ceil(df['age']).mean()
# print(df_ceil.head())
m_ceil =  np.ceil(df['age']).mean()
m_floor = np.floor(df['age']).mean()
m_trunc = np.trunc(df['age']).mean()

print(m_ceil,m_floor,m_trunc)
print(m_ceil+m_floor+m_trunc)