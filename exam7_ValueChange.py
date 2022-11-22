import pandas as pd
import numpy as np

data = pd.DataFrame([0, 1, 0, 1, 0])
data =  data.replace(0, 2)
# print(data)


df =  pd.read_csv('/workspace/goorm/basic1.csv')
print(df[df['f4']=='ESFJ'])


df['f4'] = df['f4'].replace('ESFJ', 'ISFJ')
print(df[df['f4']=='ESFJ'])


print(df[(df['f4']=='ISFJ') & (df['city']=='경기')]['age'].max())

