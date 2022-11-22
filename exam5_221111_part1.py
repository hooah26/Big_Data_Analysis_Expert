import pandas as pd
df = pd.read_csv('./basic1.csv')
'''
데이터셋(basic1.csv)의 'f5' 컬럼을 기준으로 상위 10개의 데이터를 구하고,
'f5'컬럼 10개 중 최소값으로 데이터를 대체한 후,
'age'컬럼에서 80 이상인 데이터의'f5 컬럼 평균값 구하기

'''
# print(df.head)

df = df.sort_values('f5',ascending=False)
# print(df.head)

df_min = df['f5'][:10].min()
# print(df_min)

df['f5'][:10] = df_min
# print(df.head(10))

age_avg = df[df['age']>=80]['f5'].mean()
print(age_avg)