
"""
문제1
데이터셋(basic1.csv)의 'f5' 컬럼을 기준으로 상위 10개의 데이터를 구하고,
'f5'컬럼 10개 중 최소값으로 데이터를 대체한 후,
'age'컬럼에서 80 이상인 데이터의'f5 컬럼 평균값 구하기
"""
import pandas as pd


df = pd.read_csv('./basic1.csv')
# print(df)
df.sort_values('f5', ascending=False)
print(df.head(10))
# print(df2)

f5_min = df['f5'][:10].min()
print(f5_min)

df['f5'][:10] = f5_min
# print(df1)

print(df[df['age']>80]['f5'].mean())
