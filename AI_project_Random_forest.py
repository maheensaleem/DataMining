"""
Created on Sun August  2 14:09:42 2020
@author: Maheen Saleem
Registration No: SP17-BSE-023
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix




print('\n')
df = pd.read_csv('train.csv')
X_train = df[['battery_power','four_g','int_memory','n_cores','ram','talk_time','touch_screen']]
y_train = df[['price_range']]



X_train,X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.3, random_state = 100)


model = RandomForestClassifier(n_estimators=250, 
                               bootstrap = True,
                               max_features = 'sqrt')#Typically 1/3 of data doesnot end in bootstrapped 
                                                     #dataset

                                                    #max_features:Number of features to consider
                                                    #when looking for best split
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print('\n')
print('-------------------------------------------------------------------')
