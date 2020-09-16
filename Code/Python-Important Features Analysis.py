#Random Forest

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


# use the feature importance variable to see feature importance scores.
feature_imp = pd.Series(rfcm100.feature_importances_,index=list(X.columns.values) ).sort_values(ascending=False)
feature_imp


# Gradient Boosting

import pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')

#Load data
df = pd.read_csv("4-df_preprocessed.csv")
df.shape

# Divide the data into X: predictors and Y: predicted attribute.
X = df.iloc[:, 1:]
y = df.iloc[:,0]
# Normalize X
zN = StandardScaler()
Xn = zN.fit_transform(X)
# Divide the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=.2,random_state=1234)

# Build the model
import time
start_time = time.time()
gb = GradientBoostingClassifier(max_depth=3,n_estimators=300,learning_rate=.6)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
time = round((time.time() - start_time)/60,2)
print(f"Running time is:{time} minutes.")
print('\n ** Performance Scores **')
# Build a confusion matrix and show the Classification Report
cm = confusion_matrix(y_test,y_pred)
print('\nConfusion Matrix','\n',cm)
report_classfication = classification_report(y_test,y_pred)
print('\nClassification Report','\n', report_classfication)


#Important features

# This attribute of 'feature_importance' provides the levels of significance of each attribute.
feature_names = X.columns
f_importance = gb.feature_importances_ 

features_importance = pd.DataFrame({'Features': feature_names, 'Importance':f_importance})
features_importance.sort_values(by='Importance',ascending=False, inplace=True)

print('\n** Importance values of Features **\n')   
                                                  
for _, row in features_importance.iterrows():
    print('{0:7s}: {1:.4f}'.format(row[0], row[1]))


    
# Draw a bar graph.   
plt.bar(features_importance.iloc[:,0],features_importance.iloc[:,1])
plt.title('Importance of features')
plt.xlabel('Features')
plt.ylabel('Importance Level')   
plt.show()
