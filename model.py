# -*- coding: utf-8 -*-
"""


@author: Irfan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sample = pd.read_excel(r"C:/Users/Irfan/Desktop/project69/sample data.xlsx")

sample.info()
sample.shape
sample.describe()
sample.columns

##################################
  ##### data-Preprocessing #####
##################################

##### Handling Duplicates #####
sample.duplicated().sum()

##### Renaming 2 columns #####
sample.columns
sample.rename(columns = {'Cut-off Schedule':'Cut_off_Schedule', 
                         'Cut-off time_HH_MM':'Cut_off_time_HH_MM'}, inplace = True)

sample['Test_Name'].value_counts()
sample['Sample'].value_counts()
sample['Way_Of_Storage_Of_Sample'].value_counts()
sample['Cut_off_Schedule'].value_counts()
sample['Traffic_Conditions'].value_counts()

# covert "blood" to "Blood" values
sample['Sample'] = sample['Sample']. replace(['blood'],['Blood'])

##### Missing values #####
sample.isna().sum()

##### saving cleaned data to new file
sample.to_excel('sample.xlsx')

##### Zero/Near Zero Variance #####
sample.var()    
# Mode_Of_Transport is having zero variance, so can be ignored.

###########################
sample['Reached_On_Time'].value_counts()

df_y = sample[sample['Reached_On_Time'] == 'Y']
df_n = sample[sample['Reached_On_Time'] == 'N']

df_y['Test_Name'].value_counts()
df_n['Test_Name'].value_counts()

# sorting lebalencoded Test_Name (0 to 9)
test_names = df_y['Test_Name'].unique().tolist()
test_names.sort()
test_names 
# declaring Empty dataframes
df_t = pd.DataFrame()
df_ty = pd.DataFrame()

for i in test_names:
    print('value of i:', i)
    df_t = df_y[df_y['Test_Name']==i]    
    n_count = df_n[df_n['Test_Name']==i]['Test_Name'].count()                   
    print('Count of N records for test_number ', i, ':', n_count)   
    df_t = df_t.iloc[:n_count, :]    
    df_ty = df_ty.append(df_t)
    
df_ty['Reached_On_Time'].value_counts()

sample_yn = pd.concat([df_n, df_ty])
sample_yn['Reached_On_Time'].value_counts()  # balanced sample dataset with 196 records for each N and Y

################################################################################
## Model Building ##
################################################################################ 

sample_yn.info()
sample_yn['Reached_On_Time'].value_counts()
# df_sample_yn = sample_yn.iloc[ : , [3,4,5,7,9,10,11,13,14,15,16,17,18,20]]
df_sample_yn = sample_yn.iloc[ : , [3,4,5,7,9,10,11,13,14,16,17,18,20]]
# df_sample_yn = sample_yn.iloc[ : , [3,4,5,10,11,13,16,17,18,20]]
df_sample_yn.info()
df_sample_yn['Reached_On_Time'].value_counts()

# splitting to Input/Independant and Output/Dependant Variables
X = df_sample_yn.iloc[ : , :12]  # Input/Independant variables
y = df_sample_yn.iloc[ : , 12]   # Output/Dependant Variable

# X = df_sample_yn.iloc[ : , :9]  # Input/Independant variables
# y = df_sample_yn.iloc[ : , 9]   # Output/Dependant Variable

# split into Train and test data in 70:30 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=365) 

# describes info about train and test set 
print("Number transactions X_train dataset: ", X_train.shape) 
print("Number transactions y_train dataset: ", y_train.shape) 
print("Number transactions X_test dataset: ", X_test.shape) 
print("Number transactions y_test dataset: ", y_test.shape)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

# OneHot Encoding
ohe = OneHotEncoder()
ohe.fit(X[['Test_Name', 'Sample', 'Way_Of_Storage_Of_Sample', 'Cut_off_Schedule','Traffic_Conditions']])
ohe.categories_

column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['Test_Name', 
                                       'Sample', 'Way_Of_Storage_Of_Sample','Cut_off_Schedule',
                                       'Traffic_Conditions']), remainder='passthrough')

# Hyperparameter tuning by using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid_dt = {'criterion':['gini', 'entropy'],
                 'max_depth':[2,3,4,5,6,7,8,9],
              'random_state':[1,2,3,4,5,6,7,8,9,10],
            'max_leaf_nodes':[2,3,4,5,6,7,8.9],
         'min_samples_split':[2,3,4,5,6,7,8,9]}

# Define the model with default parameters
model_grid_dt = DT(criterion='entropy', #'gini', 
              splitter='best',
              max_depth=None, 
              min_samples_split=None,
              min_samples_leaf=1, 
              min_weight_fraction_leaf=0.0, 
              max_features=None,
              random_state=None, 
              max_leaf_nodes=None,
              min_impurity_decrease=0.0, 
              class_weight=None, 
              ccp_alpha=0.0)

# Training the model with Girdsearch technique for defined set of Hyperparameters
model_dt = GridSearchCV(model_grid_dt, param_grid_dt, refit = True, verbose = 3, n_jobs = -1)

model_pipe_dt = make_pipeline(column_trans, model_dt)
model_pipe_dt.fit(X_train, y_train)

# Best set of parameters from the specified parameters
print(model_dt.best_params_)

# Prediction on Test data
y_grid_pred = model_pipe_dt.predict(X_test) 
np.mean(y_grid_pred == y_test) # Test Data Accuracy : 0.940677966101695
pd.crosstab(y_test, y_grid_pred, rownames = ['Actual'], colnames = ['Predictions'])
# print classification report for Test
print(classification_report(y_test, y_grid_pred))

# Prediction on Train data
grid_pred_tr = model_pipe_dt.predict(X_train) 
np.mean(grid_pred_tr == y_train) # Train Data Accuracy : 0.9744525547445255
pd.crosstab(y_train, grid_pred_tr, rownames = ['Actual'], colnames = ['Predictions'])
# print classification report on Train
print(classification_report(y_train, grid_pred_tr))
###############################################################################

# saving the model
# importing pickle

import pickle
pickle.dump(model_pipe_dt, open('dt_model.pkl', 'wb'))

# load the model from disk
model = pickle.load(open('dt_model.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(df_sample_yn.iloc[0:1, :13]) # 0
list_value

# list_value = pd.DataFrame(df_sample_yn.iloc[197:198, :13]) # 1
# list_value
print(model.predict(list_value))
