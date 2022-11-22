import pandas as pd
import numpy as np

df = pd.read_csv('/workspace/goorm/HousePrices/train.csv')

s1 = df['SalePrice'].skew()
k1 = df['SalePrice'].kurt()
print("왜도1:",s1)
print("첨도1:",k1)

df['SalePrice'] = np.log1p(df['SalePrice'])


s2 = df['SalePrice'].skew()
k2 = df['SalePrice'].kurt()
print("왜도2:" ,s2)
print("첨도2:" ,k2)


print(round(s1+s2+k1+k2,2))
