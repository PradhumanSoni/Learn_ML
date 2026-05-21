import numpy as np
import pandas as pd
class MyLinearReg():
    def __init__(self):
        self.m = None
        self.b = None
    
    def fit(self,x_train,y_train):
        num = np.sum((x_train.values - x_train.mean())*(y_train.values - y_train.mean()))
        den = np.sum((x_train.values -  x_train.mean())**2)
        self.m = num/den
        self.b = y_train.mean() - self.m*x_train.mean()

    def predict(self,x_test):
        return self.m*x_test + self.b
train_csv_file = input("Enter a train_csv file link here!")
test_csv_file = input("Enter a test_csv file link here!")
df_train = pd.read_csv(train_csv_file)
df_test = pd.read_csv(test_csv_file)
df_train = df_train.dropna()
df_test = df_test.dropna()
x_train = df_train.iloc[:,0]
y_train = df_train.iloc[:,-1]
x_test = df_test.iloc[:,0]
y_test = df_test.iloc[:,-1]
lr = MyLinearReg()
lr.fit(x_train,y_train)
print(f'Weightage -> {lr.m}')
print(f'Intercpet -> {lr.b}')
# inside fit() method -> we manually find each sum to find value of m using for loop
# or we can use np.sum((x_train.values - x_train.mean())*(y_train.values - y_train.mean()))
