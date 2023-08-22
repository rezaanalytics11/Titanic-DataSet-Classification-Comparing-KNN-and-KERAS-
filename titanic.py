import tensorflow as ts
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import normalize
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam
df=pd.read_csv(r'C:\Users\Ariya Rayaneh\Desktop\titanic3.csv')

df=df[['pclass','sex','age','sibsp','parch','fare','embarked','survived']]
df=df.dropna()
x=df[['pclass','sex','age','sibsp','parch','fare','embarked']]

x.embarked=x.embarked.replace(['S','C','Q'],[0,1,2])
x.sex=x.sex.replace(['female','male'],[0,1])
x=normalize(x)

y=df['survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
print(x_train.shape)
print(y_train.shape)

model0=KNeighborsClassifier(7)

model0.fit(x_train,y_train)
y_pred=model0.predict(x_test)
score=model0.score(x_test,y_test)
mse=mean_squared_error(y_test,y_pred)
print('KNN_Score_is:{}'.format(score))
print('KNN_mse_is:{}'.format(mse))

# model1=Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
#       fit_intercept=True, n_iter_no_change=5,max_iter=4,
#       n_jobs=None, penalty=None, random_state=0, shuffle=True, tol=0.001,
#       validation_fraction=0.1, verbose=0, warm_start=False)

model1=Perceptron(tol=1e-3, random_state=0,max_iter=5,shuffle=True)

model1.fit(x_train,y_train)
y_pred1=model1.predict(x_test)
score1=model1.score(x_test,y_test)
mse1=mean_squared_error(y_test,y_pred1)
print(score1)
print(mse1)
print('Perception_score_is:{}'.format(score1))
print('Perception_mse_is:{}'.format(mse1))

# model2 = tf.keras.models.Sequential([
#   tf.keras.layers.Dense(7,activation='sigmoid'),
#   tf.keras.layers.Dense(8,activation='relu'),
#   tf.keras.layers.Dense(4,activation='sigmoid'),
#   tf.keras.layers.Dense(2,activation='softmax'),
# ])
#
# sgd = SGD(lr=0.0001)
# model2.compile(optimizer='adam',
#               loss=tf.losses.sparse_categorical_crossentropy,
#               metrics=['accuracy'])
#
# model2.fit(x_train, y_train, epochs=30000,batch_size=10)
#
# model2.evaluate(x_test, y_test)
