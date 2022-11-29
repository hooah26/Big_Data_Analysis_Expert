import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def exam_data_load(df, target, id_name="", null_name=""):
    if id_name == "":
        df = df.reset_index().rename(columns={"index": "id"})
        id_name = 'id'
    else:
        id_name = id_name
    
    if null_name != "":
        df[df == null_name] = np.nan
    
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=2021)
    
    y_train = X_train[[id_name, target]]
    X_train = X_train.drop(columns=[target])

    
    y_test = X_test[[id_name, target]]
    X_test = X_test.drop(columns=[target])
    return X_train, X_test, y_train, y_test 
    
df = pd.read_csv("/workspace/goorm/diabetes.csv")
X_train, X_test, y_train, y_test = exam_data_load(df, target='Outcome')

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# print(X_train.head())
# print(y_train['Outcome'].value_counts())
# print(X_train.head())

# print(y_train.info())
# print(y_train['Outcome'].head(10))


# cols2 = X_train.columns
# for col in cols2 :
#     print(col, len(X_train[X_train[col]==0]))
#     print(col, len(X_test[X_test[col]==0]))
    
del_idx = X_train[X_train['Glucose']==0].index
# print(del_idx)
# print(X_train.shape, y_train.shape)
X_train = X_train.drop(index=del_idx, axis=0)
y_train = y_train.drop(index=del_idx, axis=0)
print(X_train.shape, y_train.shape)
cols = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
cols_mean = X_train[cols].mean()
X_train[cols] = X_train[cols].replace(0, cols_mean)


from  sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols]= scaler.fit_transform(X_test[cols])

X = X_train.drop('id', axis=1)
test =  X_test.drop('id', axis=1)

from sklearn.svm import SVC
model = SVC(random_state=2022)
model.fit(X,y_train['Outcome'])
pred = model.predict(test)
print(round(model.score(X, y_train['Outcome'])*100, 2))

output = pd.DataFrame({'idx':X_test.index,'Outcome':pred})
print(output.head())

output.to_csv('0000.scv', index=False)
check = pd.read_csv('0000.scv')
print(check.head())