from sklearn.preprocessing import StandardScaler
import pandas as pd

# df = pd.DataFrame([[0, 2], [0.4, 0.2], [1.1, 10], [11, 19], [34, 21], [6, 40]])
data = [[0, 2], [0.4, 0.2], [1.1, 10], [11, 19], [34, 21], [6, 40]]


scaler = StandardScaler()
# print(scaler.fit(data))
# print(scaler.transform(data))

df = pd.read_csv('/workspace/goorm/basic1.csv')
# print(df[['f5']])
# print(df['f5'])
df['f5']=scaler.fit_transform(df[['f5']])
print(df.head())

print(df['f5'].median())