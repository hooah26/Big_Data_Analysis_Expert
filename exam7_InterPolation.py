import pandas as pd
import numpy as np

s = pd.Series([np.nan, "퇴근후", np.nan,"딴짓", np.nan, 22, np.nan,45, np.nan, np.nan, 60])
# print(s)

a = s.fillna(method='bfill')
b = s.fillna(method='pad')
# print(a,b)

df = pd.read_csv('/workspace/goorm/basic1.csv')
df2 = df[df['f2']==1]['f1'].cumsum()
print(df2)
df2 = df2.fillna(method='bfill')
print(df2)
print(df2.mean())