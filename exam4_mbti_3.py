
"""
문제3
데이터셋(basic1.csv)의 'age'컬럼의 이상치를 더하시오!
단, 평균으로부터 '표준편차*1.5'를 벗어나는 영역을 이상치라고 판단함
"""


import pandas as pd
import numpy as np

df = pd.read_csv('./basic1.csv')
# print(df)

std = df['age'].std()
mean = df['age'].mean()
# print(std, mean)

min = mean-std*1.5
max = mean+std*1.5
print(min, max)

outlier = df[(df['age']<min)|(df['age']>max)]['age'].sum()
print(outlier)