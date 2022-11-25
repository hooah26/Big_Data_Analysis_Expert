import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/basic1.csv')
# print(df.head)
# print(df.isna().sum())


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['f5_1'] = scaler.fit_transform(df[['f5']])

# print(df.head())

lower = df['f5_1'].quantile(0.05)
print(lower)

uper = df['f5_1'].quantile(0.95)
print(uper)

print(lower+uper)

