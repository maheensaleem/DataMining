"""
Created on Sun August  2 14:09:42 2020
@author: Maheen Saleem
Registration No: SP17-BSE-023
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np





print('\n')
df = pd.read_csv('train.csv')

X = np.array(df[['battery_power','four_g','int_memory','n_cores','ram','talk_time','touch_screen']])
y = np.array(df[['price_range']])

AX = np.array(df[['battery_power','four_g','int_memory','n_cores','ram','talk_time','touch_screen']])





def clf_gini_kfold(X, y):
    
    
    avg = [0 for i in range(0,5)]
    j=0
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
                               max_depth=None, min_samples_leaf=1)#A split point any depth will be 
                                                                  #considered if it has 1 node on left and
                                                                  #right branches
                                                             

    cv = KFold(n_splits=5, random_state=42, shuffle=False)
    

    for train_index, test_index in cv.split(X):
     
        
         X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
         clf_gini.fit(X_train, y_train)
         y_pred = clf_gini.predict(X_test) 
         accuracy = accuracy_score(y_test,y_pred)*100
         avg[j]=accuracy
         j=j+1
         print(accuracy)
         
         
    print('\nAverage Gini:',np.mean(avg))
    
    return np.mean(avg) 
    
    
    
    
def clf_entropy_kfold(X,y):
    
    avg = [0 for i in range(0,4)]
    j=0
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                         max_depth=None, min_samples_leaf=1)
    
    
    cv = KFold(n_splits=4, random_state=42, shuffle=False)



    for train_index, test_index in cv.split(X):
     
         X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
         clf_entropy.fit(X_train, y_train)
         y_pred = clf_entropy.predict(X_test) 
         accuracy = accuracy_score(y_test,y_pred)*100
         avg[j]=accuracy
         j=j+1
         print(accuracy)
         
    print('\nAverage Entropy:',np.mean(avg))
    

    return np.mean(avg)


def random_forest_kfold(X,y,n_estimators):
    
    avg = [0 for i in range(0,4)]
    j=0
    
    cv = KFold(n_splits=4, random_state=42, shuffle=False)
    
    model = RandomForestClassifier(n_estimators, 
                               bootstrap = True,
                               max_features = 'sqrt')#Typically 1/3 of data doesnot end in bootstrapped 
                                                     #dataset

                                                    #max_features:Number of features to consider
                                                    #when looking for best split

    
    
    for train_index, test_index in cv.split(X):
 
         X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
         model.fit(X_train,y_train)
         y_pred = model.predict(X_test)
         accuracy = accuracy_score(y_test,y_pred)*100
         avg[j]=accuracy
         j=j+1
         print(accuracy)
         
    print('\nAverage Random Forest:',np.mean(avg))
    
    return np.mean(avg)



