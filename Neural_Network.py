#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:52:46 2024

@author: arnaudcruchaudet
"""

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X=cancer['data']
y=cancer['target']

X_train,X_test,y_train,y_test = train_test_split(X,y)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#MLP with default options, with 2 hidden layers of 30 and 15 perceptrons respectively
mlp = MLPClassifier(hidden_layer_sizes=(30,15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) #55=vrai négatifs, 3=faux négatif, 2=faux positifs, 83=vrai positifs
print(classification_report(y_test,predictions)) 


#MLP with main option to choose
mlp = MLPClassifier(activation = 'logistic', solver= 'sgd', alpha=0.000001, learning_rate='adaptive',  hidden_layer_sizes=(30,15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 


mlp = MLPClassifier(activation = 'relu', solver= 'sgd', alpha=0.000001, learning_rate='adaptive',  hidden_layer_sizes=(30,15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 



mlp = MLPClassifier(activation = 'logistic', solver= 'sgd', learning_rate='adaptive',  hidden_layer_sizes=(30,15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) #99% de succès



#MLP with Strength of the L2 regularization term fixed (Ridge Regression) - The larger the value of alpha, the less variance your model will exhibit- default=0.0001
mlp = MLPClassifier(activation = 'logistic', solver= 'sgd', alpha=0.000001, learning_rate='adaptive',  hidden_layer_sizes=(30,15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 


mlp = MLPClassifier(activation = 'relu', solver= 'sgd', alpha=0.000001, learning_rate='constant',  hidden_layer_sizes=(30,15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 


mlp = MLPClassifier(activation = 'tanh', solver= 'sgd', learning_rate='adaptive',  hidden_layer_sizes=(30,15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 

mlp = MLPClassifier(activation = 'tanh', solver= 'sgd', learning_rate='invscaling',  hidden_layer_sizes=(30,15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 


mlp = MLPClassifier(activation = 'relu', solver= 'sgd', alpha=0.000001, learning_rate='adaptive',  hidden_layer_sizes=(30,15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 



mlp = MLPClassifier(activation = 'relu', solver= 'adam', alpha=0.000001,  hidden_layer_sizes=(30,15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions))



mlp = MLPClassifier(activation = 'relu', solver= 'adam', alpha=0.005,  hidden_layer_sizes=(30,15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 


mlp = MLPClassifier(activation = 'relu', solver= 'adam', alpha=0.005,  hidden_layer_sizes=(10,10,10), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 


mlp = MLPClassifier(activation = 'relu', solver= 'adam', alpha=0.005,  hidden_layer_sizes=(10,10,10,20), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 

mlp = MLPClassifier(activation = 'relu', solver= 'adam', alpha=0.005,  hidden_layer_sizes=(10,10,10,20, 15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 


mlp = MLPClassifier(activation = 'relu', solver= 'sgd', alpha=0.005,  hidden_layer_sizes=(10,10,10,20, 15), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions)) 

mlp = MLPClassifier(activation = 'relu', solver= 'adam', alpha=0.000001,  hidden_layer_sizes=(40,30), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions))


mlp = MLPClassifier(activation = 'relu', solver= 'adam', alpha=0.000001,  hidden_layer_sizes=(10,10), max_iter=10000)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print(y_test)
print(predictions)
print(confusion_matrix(y_test,predictions)) 
print(classification_report(y_test,predictions))


#### Regression task 
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
XX = housing.data[:,:]  
y = housing.target
XX2 = StandardScaler().fit_transform(XX)
X_train,X_test,y_train,y_test = train_test_split(XX2,y)

mlp = MLPRegressor(activation = 'relu', solver= 'adam', alpha=0.000001,  hidden_layer_sizes=(50,40,30,20), max_iter=10000)
pred = mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
pred.predict(X_test)
print(y_test)
print(predictions)
print (pred.score(X_test, y_test))