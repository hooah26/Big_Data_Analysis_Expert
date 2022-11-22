'''
데이터셋(basic1.csv)의 'age'컬럼의 이상치를 더하시오!
단, 평균으로부터 '표준편차*1.5'를 벗어나는 영역을 이상치라고 판단함
'''

import pandas as pd
import numpy as np

df = pd.read_csv('./basic1.csv')
# print(df.head)

mean =  df['age'].mean()
std = df['age'].std()
# print(mean,std)

min_out = mean-std*1.5
max_out = mean+std*1.5
# print(min_out,max_out)

outlier = df[(df['age']>max_out) | (df['age']<min_out)]['age'].sum()
print(outlier)