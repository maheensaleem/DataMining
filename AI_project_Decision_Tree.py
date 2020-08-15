"""
Created on Sun August  2 14:09:42 2020
@author: Maheen Saleem
Registration No: SP17-BSE-023
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

print('\n')
df = pd.read_csv('train.csv')
X_train = df[['battery_power','four_g','int_memory','n_cores','ram','talk_time','touch_screen']]
y_train = df[['price_range']]


X_train,X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.3, random_state = 100)




def clf_gini(battery, four_g, int_mem, n_cores, ram ,talk_time, touch_screen,X_train, y_train, X_test):
    
    
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=None, min_samples_leaf=1) #A split point any depth will be 
                                                                   #considered if it has 1 node on left and
                                                                   #right branches
    clf_gini.fit(X_train, y_train)
    
    
    return clf_gini.predict([[battery,four_g,int_mem, n_cores,ram,talk_time,touch_screen]]), clf_gini.predict(X_test[['battery_power','four_g','int_memory','n_cores','ram','talk_time','touch_screen']])
    
    


def clf_entropy(battery, four_g, int_mem, n_cores, ram ,talk_time, touch_screen,X_train, y_train, X_test):
    
    
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                         max_depth=None, min_samples_leaf=1)
    clf_entropy.fit(X_train, y_train)

    
    return clf_entropy.predict([[battery, four_g, int_mem, n_cores, ram ,talk_time, touch_screen]]),  clf_entropy.predict(X_test[['battery_power','four_g','int_memory','n_cores','ram','talk_time','touch_screen']])