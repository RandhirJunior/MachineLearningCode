# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:36:03 2020

@author: Randhirs
"""

# Data preprocessing-----------
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#---read csv fle using pandas dataset
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:,3].values

# Taking care of missing data
# we are not using it in our templete
"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

"""

# Encoding categorical data
"""
from sklearn.preprocessing  import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencode=OneHotEncoder(categorical_features=[0])
X=onehotencode.fit_transform(X).toarray()
"""

#for purchase items
"""
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)
"""


# Spliting the dataset into the Training set and test set
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Features Scaling
""" 
from sklearn.preprocessing import  StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
"""
