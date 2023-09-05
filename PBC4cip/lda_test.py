import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def split_data(train, test):
    X_train = train.iloc[:,  0:train.shape[1]-1]
    y_train =  train.iloc[:, train.shape[1]-1 : train.shape[1]]

    X_test = test.iloc[:,  0:test.shape[1]-1]
    y_test =  test.iloc[:, test.shape[1]-1 : test.shape[1]]

    return X_train.to_numpy(), y_train.to_numpy().ravel(), X_test.to_numpy(), y_test.to_numpy().ravel()

def import_data(trainFile, testFile):
    train = pd.read_csv(trainFile) 
    test = pd.read_csv(testFile)
    return train, test

current_location = os.path.dirname(os.path.abspath(__file__))
trainFile = current_location + '\\example\\train.csv'
testFile = current_location + '\\example\\test.csv'
train, test = import_data(trainFile, testFile)
X_train, y_train, X_test, y_test = split_data(train, test)

lda = LDA(solver='eigen')
lda = lda.fit(X_train, y_train)

coef = lda.coef_

projections = X_train * lda.coef_
#X_projected = lda.transform(X_train)

print(projections.shape)

print(np.sum(projections, axis= 1))
#print(lda.scalings_)
print(coef)