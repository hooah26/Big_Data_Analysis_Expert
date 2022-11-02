import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = sns.load_dataset('mpg')
X_train, X_test, Y_train, Y_test = train_test_split(df, df['mpg'], test_size=0.2, random_state=42)
X_train = X_train.drop(['mpg'], axis=1)
X_test = X_test.drop(['mpg'], axis=1)

#사용자 코딩
# print(X_train.head())
# print(X_train.isna().sum())

#1.결측치 제거
X_train['horsepower'] = X_train['horsepower'].fillna(X_train['horsepower'].median())
X_test['horsepower'] = X_test['horsepower'].fillna(X_test['horsepower'].median())
# print(X_train.isna().sum())
# print(X_test.isna().sum())

#2.라벨인코더
from sklearn.preprocessing import LabelEncoder
label = ['origin','name']
X_train[label] =  X_train[label].apply(LabelEncoder().fit_transform)
X_test[label] =  X_test[label].apply(LabelEncoder().fit_transform)

# print(X_train.head())

#3.카테고리 데이터 타입 변경 및 더미
category = ['origin']
for i in category :
    X_train[i] = X_train[i].astype('category')
    X_test[i] = X_test[i].astype('category')
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
# print(X_train.head())

#4.파생변수 만들기
X_train['horsepower_qcut'] = pd.qcut(X_train['horsepower'], 5, labels=False)
X_test['horsepower_qcut'] = pd.qcut(X_test['horsepower'], 5, labels=False)
# print(X_train.head())

#5.스케일
from sklearn.preprocessing import MinMaxScaler
scaler = ['displacement','horsepower','weight']
min = MinMaxScaler()
min.fit(X_train[scaler])
min.fit(X_test[scaler])

X_train[scaler] = min.transform(X_train[scaler])
X_test[scaler] = min.transform(X_test[scaler])
# print(X_train.head())

#6.데이터 분리
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
# print(X_train.shape)
# print(X_test.shape)

#7.모형 학습 regression 분리
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train, Y_train)
pred1 = model1.predict(X_valid)


from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor()
model2.fit(X_train, Y_train)
pred2 = model2.predict(X_valid)


#8.앙상블(스태킹)
from sklearn.ensemble import StackingRegressor
estimators = [('rl', model1),('rf',model2)]
model3 = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor())
model3.fit(X_train, Y_train)
pred3 = model3.predict(X_valid)
print(pred3)

#9.모형평가
from sklearn.metrics import mean_squared_error
print('선형회귀 MSE', mean_squared_error(Y_valid,pred1))
print('랜포 MSE', mean_squared_error(Y_valid,pred2))
print('스태킹 MSE', mean_squared_error(Y_valid,pred3))

print('선형회귀 RMSE', np.sqrt(mean_squared_error(Y_valid,pred1)))
print('랜포 RMSE', np.sqrt(mean_squared_error(Y_valid,pred2)))
print('스태킹 RMSE', np.sqrt(mean_squared_error(Y_valid,pred3)))

#10.하이퍼 파라미터 튜닝
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[50,100], 'max_depth':[4,6]}
model4 = RandomForestRegressor()
clf = GridSearchCV(estimator=model4, param_grid=parameters, cv=3)
clf.fit(X_train, Y_train)
# print('최적의파라미터', clf.best_params_)

#11.파일저장
result = pd.DataFrame(model2.predict(X_test))
result = result.iloc[:,0]
pd.DataFrame({'id':X_test.index, 'result':result}).to_csv('00400.csv', index=False)
check = pd.read_csv('00400.csv')
print(check.head())