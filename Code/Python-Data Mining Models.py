#Bayesian for Preprocessed dataset
# Load dataset
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection  import train_test_split
from sklearn import metrics
import statsmodels.api as sm
from mixed_naive_bayes import MixedNB
from sklearn.preprocessing import LabelEncoder
pro2 = pd.read_csv("4-df_preprocessed.csv", sep=',', header=0)

#Divide the dataset into y and X
y = pro2['Severity']
X = pro2.iloc[:,1:]

# Change categorical variables into numerical variables
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X.iloc[:,[0,1,2,3,4,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,28,30,31,32,33,34,35]] = X.iloc[:,[0,1,2,3,4,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,28,30,31,32,33,34,35]].apply(LabelEncoder().fit_transform)

# Split the dataset into training dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.2,random_state=1234, stratify=y)

# Build a Bayesian Classification Model and predict the type using the test data.
gnb = MixedNB(categorical_features=[0,1,2,3,4,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,28,30,31,32,33,34,35])
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# Calculate the accuracy
accuracy = gnb.score(X_test, y_test)
print('Accuracy: {0:.2f}'.format(accuracy))

# Build a confusion matrix
cm = metrics.confusion_matrix(y_test,y_pred)
print(metrics.classification_report(y_test,y_pred))


#Bayesian for PCA dataset
# Load dataset
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection  import train_test_split
from sklearn import metrics
import statsmodels.api as sm
from mixed_naive_bayes import MixedNB
from sklearn.preprocessing import LabelEncoder
pro3 = pd.read_csv("5-df_PCA_1.csv", sep=',', header=0)

#Divide the dataset into y and X
y = pro3['0']
X = pro3.iloc[:,1:]

# Split the dataset into training dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.2,random_state=1234, stratify=y)

# Build a Bayesian Classification Model and predict the type using the test data.
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# Calculate the accuracy
accuracy = gnb.score(X_test, y_test)
print('Accuracy: {0:.2f}'.format(accuracy))

# Build a confusion matrix
cm = metrics.confusion_matrix(y_test,y_pred)
print(metrics.classification_report(y_test,y_pred))


# K-Nearest Neighbors for Preprocessed dataset

#Load the data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, datasets
import matplotlib.pyplot as plt
pro_knn2 = pd.read_csv("4-df_preprocessed.csv", sep=',', header=0)

#Divide the dataset into y and X
y = pro_knn2['Severity']
X = pro_knn2.iloc[:,1:]

#Standardize the X
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the dataset into training dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.1,random_state=1234, stratify=y)

#Build the model
knn = KNeighborsClassifier(n_neighbors = 20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Print the performance scores.
print('\n ** Performance Scores **')

# Calculate Accuracy
accuracy = knn.score(X_test, y_test)
print('Accuracy: ', accuracy)

# Build a confusion matrix and show the Classification Report
cm = metrics.confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix','\n',cm)
print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred))


# K-Nearest Neighbors for PCA dataset

#Load the data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, datasets
import matplotlib.pyplot as plt
pro_knn2 = pd.read_csv("5-df_PCA_1.csv", sep=',', header=0)

#Divide the dataset into y and X
y = pro_knn2['Severity']
X = pro_knn2.iloc[:,1:]

#Standardize the X
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the dataset into training dataset and test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.1,random_state=1234, stratify=y)

#Build the model
knn = KNeighborsClassifier(n_neighbors = 20)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Print the performance scores.
print('\n ** Performance Scores **')

# Calculate Accuracy
accuracy= knn.score(X_test, y_test)
print('Accuracy: ', accuracy)

# Build a confusion matrix and show the Classification Report
cm = metrics.confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix','\n',cm)
print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred))

# Decision Tree for Preprocessed dataset

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics, datasets, tree
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# load the preprocessed data set
DF2 = pd.read_csv("4-df_preprocessed.csv", sep=',', header=0)
print(DF2.info())
DF2.head()
DF2.describe()

# Divide the data into X: predictors and y: predicted attribute.

X = DF2.iloc[:,1 :] # The first column[0] has 'severity' to be predicted.
y = DF2.iloc[:,0]
X.shape
y.shape

# Decision Tree Classification

# Split the data into training and testing subsets.
X_train, X_test = train_test_split(X)
# Set 20% data for testing 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.2,random_state=1234, stratify=y)

# Create a model (object) for classification
dt2_1 = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=3, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')

# Build a decision tree
dt2_1.fit(X_train, y_train)
y_pred = dt2_1.predict(X_test)

# Calculate accuracy
accuracy = dt2_1.score(X_test, y_test)
print('Accuracy: {0:.2f}'.format(accuracy))

# Build a confusion matrix and show the Classification Report
cm_1 = metrics.confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix','\n',cm_1)
print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred))


# Create a decision tree plot.
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,4), dpi=300)
tree.plot_tree(dt2_1)

# Decision Tree Regression

# Split the data into training and testing subsets.
X_train, X_test = train_test_split(X)
# Alternatively, you can set the parameter values differently from the default ones.
X_train, X_test = train_test_split(X, test_size =.2,random_state=1234)
y_train, y_test = train_test_split(y, test_size=.2, random_state=1234)

# Create a model (object-'dtm') from the class 'DecisionTreeRegressor'
# with a random seed value to be consistent with many executions.
dtm1 = DecisionTreeRegressor(max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=10,                             min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False,                              random_state=100, splitter='best')

# Build a decision tree and predict the values using the test data.
dtm1.fit(X_train, y_train)
y_pred = dtm1.predict(X_test)

# Calculate MSE, RMSE, and R^2 for errors.
# Compare these values with the ones from other models.

mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
rsquared = dtm1.score(X_test, y_test)
print('\n', '**Evaluation of Errors**')
print (' mse: ', mse,'\n','rmse:', rmse)
print (' R^2: ', rsquared)


# Decision Tree for PCA dataset

#Load dataset
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics, datasets, tree
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
DF3 = pd.read_csv("5-df_PCA_1.csv", sep=',', header=0)
print(DF3.info())

# Divide the data into X: predictors and Y: predicted attribute.
X = DF3.iloc[:,1 :] # The first column[0] has 'severity' to be predicted.
y = DF3.iloc[:,0]
X.shape
y.shape

# Split the data into training and testing subsets.
X_train, X_test = train_test_split(X)
# Set 20% data for testing 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.2,random_state=1234, stratify=y)

# Create a model (object) for classification
dt3_1 = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=3, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=100, splitter='best')

# Build a decision tree
dt3_1.fit(X_train, y_train)
y_pred = dt3_1.predict(X_test)

# Calculate accuracy
accuracy = dt3_1.score(X_test, y_test)
print('Accuracy: {0:.2f}'.format(accuracy))

# Build a confusion matrix and show the Classification Report
cm_1 = metrics.confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix','\n',cm)
print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred))


# Create a decision tree plot.
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,4), dpi=300)
tree.plot_tree(dt3_1)

# Decision Tree Regression
# Split the data into training and testing subsets.
X_train, X_test = train_test_split(X)
# Alternatively, you can set the parameter values differently from the default ones.
X_train, X_test = train_test_split(X, test_size =.2,random_state=1234)
y_train, y_test = train_test_split(y, test_size=.2, random_state=1234)

# Create a model (object-'dtm') from the class 'DecisionTreeRegressor'
# with a random seed value to be consistent with many executions.
dtm1 = DecisionTreeRegressor(max_depth=10, max_features=None, max_leaf_nodes=None, min_samples_leaf=10,                             min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False,                              random_state=100, splitter='best')

# Build a decision tree and predict the values using the test data.
dtm1.fit(X_train, y_train)
y_pred = dtm1.predict(X_test)

# Calculate MSE, RMSE, and R^2 for errors.
# Compare these values with the ones from other models.
mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
rsquared = dtm1.score(X_test, y_test)
print('\n', '**Evaluation of Errors**')
print (' mse: ', mse,'\n','rmse:', rmse)
print (' R^2: ', rsquared)

# Random Forest for Preprocessed dataset

# Random Forest 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing

# Read Data after pre-processing is applied, including outlier and correlation analysis.
data = pd.read_csv('4-df_preprocessed.csv')
data.shape

# Divide the data into X: predictors and Y: predicted attribute.

X = data.iloc[:,1:] # The first column [0] has 'severity' to be predicted.
y = data.iloc[:,0]

# Set 20% data for testing.
X_train, X_test = train_test_split(X, test_size =.2,random_state=1234)
y_train, y_test = train_test_split(y, test_size=.2, random_state=1234)

# Numerical Prediction
# Create a model (object-'rfrm') from the class 'RandomForestRegressor'
# with a random seed value to be consistent with many executions.
rfrm = RandomForestRegressor(random_state=1234)

# Build a Random Forest and predict the severity using the test data.
rfrm.fit(X_train, y_train)
y_pred = rfrm.predict(X_test)

# Calculate MSE, RMSE, and R^2 for errors.
mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
rsquared = rfrm.score(X_test, y_test)
print('\n', '**Evaluation of Errors**')
print (' mse: ', mse,'\n','rmse:', rmse)
print (' R^2: ', rsquared)
Output:
**Evaluation of Errors**
 mse:  0.20709379952898532 
 rmse: 0.4550755975977896
 R^2:  0.2685458130478465

# Rounding prediction to int for confusion matrix
y_pred_round=np.rint(y_pred)
# Print the performance scores.
print('\n ** Performance Scores **')
# Build a confusion matrix and show the Classification Report
cm = metrics.confusion_matrix(y_test,y_pred_round)
print('\nConfusion Matrix','\n',cm)
print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred_round))

# Classification - 100 trees

# Create a model (object) for classification
rfcm100 = RandomForestClassifier(n_estimators=100)

# Build a random forest classification model
rfcm100.fit(X_train, y_train)
y_pred = rfcm100.predict(X_test)

# Print the performance scores.
print('\n ** Performance Scores **')
# Calculate accuracy
accuracy = rfcm100.score(X_test, y_test)
print('Accuracy: {0:.2f}'.format(accuracy))
# Build a confusion matrix and show the Classification Report
cm = metrics.confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix','\n',cm)
print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred))

#Random Forest for PCA dataset

# Random Forest 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing

# Load data
data = pd.read_csv('5-df_PCA_1.csv')
data.shape

# Divide the data into X: predictors and Y: predicted attribute.
X = data.iloc[:,1:] # The first column [0] has 'severity' to be predicted.
y = data.iloc[:,0]

# Set 20% data for testing.
X_train, X_test = train_test_split(X, test_size =.2,random_state=1234)
y_train, y_test = train_test_split(y, test_size=.2, random_state=1234)

# Numerical Prediction

# Create a model (object-'rfrm') from the class 'RandomForestRegressor'
# with a random seed value to be consistent with many executions.
rfrm = RandomForestRegressor(random_state=1234)

# Build a Random Forest and predict the severity using the test data.
rfrm.fit(X_train, y_train)
y_pred = rfrm.predict(X_test)

# Calculate MSE, RMSE, and R^2 for errors.
mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
rsquared = rfrm.score(X_test, y_test)
print('\n', '**Evaluation of Errors**')
print (' mse: ', mse,'\n','rmse:', rmse)
print (' R^2: ', rsquared)

Output:
**Evaluation of Errors**
 mse:  0.22779782141468705 
 rmse: 0.4772817002721632
 R^2:  0.19957056361157732

# Rounding prediction to int for confusion matrix
y_pred_round=np.rint(y_pred)
# Print the performance scores.
print('\n ** Performance Scores **')
# Build a confusion matrix and show the Classification Report
cm = metrics.confusion_matrix(y_test,y_pred_round)
print('\nConfusion Matrix','\n',cm)
print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred_round))

# Classification - 100 trees

# Create a model (object) for classification
rfcm100 = RandomForestClassifier(n_estimators=100)

# Build a random forest classification model
rfcm100.fit(X_train, y_train)
y_pred = rfcm100.predict(X_test)

# Print the performance scores.
print('\n ** Performance Scores **')

# Calculate accuracy
accuracy = rfcm100.score(X_test, y_test)
print('Accuracy: {0:.2f}'.format(accuracy))

# Build a confusion matrix and show the Classification Report
cm = metrics.confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix','\n',cm)
print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred))


# Neural Networks for Preprocessed dataset

import time
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import metrics, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Neural Network Classification

# setup
start_time = time.time()
print ("DATA SET 2: POST-OUTLIER ANALYSIS")

# Read the data 
accidents = pd.read_csv("4-df_preprocessed.csv")
print(accidents.describe())

#get a random sample before running with full sample
accidents = accidents.sample(frac=0.1, random_state = 1234)

X = accidents[accidents.columns.difference(["Severity"])]
y = accidents["Severity"]

# Normalize the data
Xn = scale(X)

# Create training and test set
Xn_train, Xn_test = train_test_split(Xn, test_size =.25,random_state=1234, stratify=y)
y_train, y_test = train_test_split(y, test_size=.25, random_state=1234, stratify=y)
 
# Selection of parameter values

nnm = MLPClassifier()

params = {'hidden_layer_sizes':[(20,), (100,),(100,50)],'activation':
          ['logistic', 'tanh','relu'], 'max_iter': [1000,2000,3000,4000,5000]}

grid_search = GridSearchCV(estimator= nnm, param_grid= params, scoring='accuracy')
grid_search.fit(Xn_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)

Output:
0.754267140262

# Create a model for regression using the best params recommended

nnm = MLPClassifier(hidden_layer_sizes=(20,), max_iter=5000,activation='logistic')

# Make predictions
nnm.fit(Xn_train, y_train)
y_pred = nnm.predict(Xn_test)

#  Performance Scores
print('\n ** Performance Scores **')
# Build a confusion matrix and show the Classification Report
cm = metrics.confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix','\n',cm)
print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred))

#Regression

# Read the data 
accidents = pd.read_csv("4-df_preprocessed.csv")
print(accidents.describe())

#get a random sample before running with full sample
accidents = accidents.sample(frac=0.1, random_state = 1234)

X = accidents[accidents.columns.difference(["Severity"])]
y = accidents["Severity"]

# Normalize the data
Xn = scale(X)

# Create training and test set
Xn_train, Xn_test = train_test_split(Xn, test_size =.25,random_state=1234, stratify=y)
y_train, y_test = train_test_split(y, test_size=.25, random_state=1234, stratify=y)

# Selection of parameter values

nnm = MLPRegressor()

params = {'hidden_layer_sizes':[(20,), (100,),(100,50)],'activation':
          ['logistic', 'tanh','relu'], 'max_iter': [1000,2000,3000,4000,5000]}

grid_search = GridSearchCV(estimator= nnm, param_grid= params, scoring='r2')
grid_search.fit(Xn_train, y_train)

print(grid_search.best_params_) # can be different at every execution.
print(grid_search.best_score_)

# Neural Networks for PCA dataset

import time
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import metrics, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Classification

print ("DATA SET 3: PCA DATA")

# Read the data 
accidents = pd.read_csv("5-df_PCA_1.csv")
print(accidents.describe())

#get a random sample before running with full sample
accidents = accidents.sample(frac=0.1, random_state = 1234)

X = accidents[accidents.columns.difference(["0"])]
y = accidents["0"]

# Normalize the data
Xn = scale(X)

# Create training and test set
Xn_train, Xn_test = train_test_split(Xn, test_size =.25,random_state=1234, stratify=y)
y_train, y_test = train_test_split(y, test_size=.25, random_state=1234, stratify=y)

# Selection of parameter values

nnm = MLPClassifier()

params = {'hidden_layer_sizes':[(20,), (100,),(100,50)],'activation':
          ['logistic', 'tanh','relu'], 'max_iter': [1000,2000,3000,4000,5000]}

grid_search = GridSearchCV(estimator= nnm, param_grid= params, scoring='accuracy')
grid_search.fit(Xn_train, y_train)

print(grid_search.best_params_)
print(grid_search.best_score_)


# Create a model for regression using the best params recommended.

nnm = MLPClassifier(hidden_layer_sizes=(100,), max_iter=4000,activation='relu')

# Make predictions
nnm.fit(Xn_train, y_train)
y_pred = nnm.predict(Xn_test)

#  Performance Scores

print('\n ** Performance Scores **')
# Build a confusion matrix and show the Classification Report
cm = metrics.confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix','\n',cm)
print('\nClassification Report','\n',metrics.classification_report(y_test,y_pred))


# Regression

# Read the data 
accidents = pd.read_csv("5-df_PCA_1.csv")
print(accidents.describe())

#get a random sample before running with full sample
accidents = accidents.sample(frac=0.1, random_state = 1234)

X = accidents[accidents.columns.difference(["0"])]
y = accidents["0"]

# Normalize the data
Xn = scale(X)

# Create training and test set
Xn_train, Xn_test = train_test_split(Xn, test_size =.25,random_state=1234, stratify=y)
y_train, y_test = train_test_split(y, test_size=.25, random_state=1234, stratify=y)

# Selection of parameter values

nnm = MLPRegressor()

params = {'hidden_layer_sizes':[(20,), (100,),(100,50)],'activation':
          ['logistic', 'tanh','relu'], 'max_iter': [1000,2000,3000,4000,5000]}

grid_search = GridSearchCV(estimator= nnm, param_grid= params, scoring='r2')
grid_search.fit(Xn_train, y_train)

print(grid_search.best_params_) # These values can be different at every execution.
print(grid_search.best_score_)


# Gradient Boosting for Preprocessed dataset

import pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')
n = 100000

# Gradient Boosting for Classification

# Data after pre-processing 
df2 = pd.read_csv("4-df_preprocessed.csv")
df_sample2 = df2.sample(n, random_state=1234)
df_sample2.shape

# Divide the data into X: predictors and Y: predicted attribute.
X = df_sample2.iloc[:, 1:]
y = df_sample2.iloc[:,0]
# Normalize X
zN = StandardScaler()
Xn = zN.fit_transform(X)
# Divide the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=.2,random_state=1234)

# Selection of parameter values for classification model
from sklearn.model_selection import GridSearchCV
import time
start_time = time.time()
my_params = {'max_depth': [2,3,5],'learning_rate':[.3,.5,.7], 
             'n_estimators':[100,200,300]}
gs = GridSearchCV(estimator=GradientBoostingClassifier(),
                      param_grid=my_params, scoring='accuracy')
gs.fit(X_train, y_train)
print(gs.best_params_)
time = (time.time() - start_time)/60
print(time)

# Create a model
import time
start_time = time.time()
gb = GradientBoostingClassifier(max_depth=3,n_estimators=300,learning_rate=.6)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
time = (time.time() - start_time)/60
print(f"Running time is:'{time}'minutes.")

print('\n ** Performance Scores **')
# Build a confusion matrix and show the Classification Report
cm = confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix','\n',cm)
report_sample2 = classification_report(y_test,y_pred)
print('\nClassification Report','\n', report_sample2)
accuracy2 = accuracy_score(y_test, y_pred)


# Gradient Boosting for Regression

df2 = pd.read_csv("4-df_preprocessed.csv")
df_sample2 = df2.sample(n,random_state=1234)
df_sample2.shape

# Divide the data into X: predictors and Y: predicted attribute.
X = df_sample2.iloc[:, 1:]
y = df_sample2.iloc[:,0]
# Normalize X
zN = StandardScaler()
Xn = zN.fit_transform(X)
# Divide the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=.2,random_state=1234)

# Selection of parameter values for regression model
from sklearn.model_selection import GridSearchCV
import time
start_time = time.time()
my_params = {'max_depth': [3,5,7],'learning_rate':[.3,.5,.7], 
             'n_estimators':[200,300,400]}
gs = GridSearchCV(estimator=GradientBoostingRegressor(),
                      param_grid=my_params, scoring='r2')
gs.fit(X_train, y_train)
print(gs.best_params_)
time = (time.time() - start_time)/60
print(time)

# Create a model
import time
start_time = time.time()
gb = GradientBoostingRegressor(max_depth=3,n_estimators=500,learning_rate=.4)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
rsquare2 = r2_score(y_test, y_pred)
time = (time.time() - start_time)/60
print(f"Running time is:{time}minutes.")
print('\n', '**Evaluation of Errors**')
print (' mse: ', mse,'\n','rmse:', rmse)
print(f'R^2: {rsquare2}.')


# Gradient Boosting for PCA dataset

import pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
get_ipython().run_line_magic('matplotlib', 'inline')
n = 100000

# Gradient Boosting for Classification

#Load the data
df3 = pd.read_csv("5-df_PCA_1.csv")
df_sample3 = df3.sample(n,random_state=1234)
df_sample3.shape

# Divide the data into X: predictors and Y: predicted attribute.
X = df_sample3.iloc[:, 1:]
y = df_sample3.iloc[:,0]
# Divide the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=1234)

# Create a model
import time 
start_time = time.time()
gb = GradientBoostingClassifier(max_depth=3,n_estimators=200,learning_rate=.5)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
time = (time.time() - start_time)/60
print(f"Running time is:'{time}'minutes.")
print('\n ** Performance Scores **')
# Build a confusion matrix and show the Classification Report
cm = confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix','\n',cm)
report_sample3 = classification_report(y_test,y_pred)
print('\nClassification Report','\n', report_sample3)
accuracy3 = accuracy_score(y_test, y_pred)

# Gradient Boosting for Regression

#Load data
df3 = pd.read_csv("5-df_PCA_1.csv")
df_sample3 = df3.sample(n,random_state=1234)
df_sample3.shape

# Divide the data into X: predictors and Y: predicted attribute.
X = df_sample3.iloc[:, 1:]
y = df_sample3.iloc[:,0]
# Divide the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=1234)

# Create a model
import time
start_time = time.time()
gb = GradientBoostingRegressor(max_depth=3,n_estimators=400,learning_rate=.4)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test,y_pred))
rsquare3 = r2_score(y_test, y_pred)
time = (time.time() - start_time)/60
print(f"Running time is:{time}minutes.")
print('\n', '**Evaluation of Errors**')
print (' mse: ', mse,'\n','rmse:', rmse)
print(f'R^2: {rsquare3}.')
