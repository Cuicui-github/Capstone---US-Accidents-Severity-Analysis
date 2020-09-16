
# K-mean clustering
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.cluster import KMeans
# Load Preprocessed dataset
df = pd.read_csv("4-df_preprocessed.csv")
df.shape

# Sample the dataset

df_sample = df.sample(frac = 0.1,random_state = 1)
df_sample.shape
X = df_sample.iloc[:, 1:]
y = df_sample.iloc[:,0]

# Standardize the data

df_n = minmax_scale(X)
df_n

# Let's try to find the k value using the elbow method.

import time
start_time = time.time()
ssd = [] # Initialize the list for inertia values - sum of squared distances
for i in range(2,20):
    km = KMeans(n_clusters=i, random_state=1234)
    km.fit(df_n)
    ssd.append(km.inertia_)
time = (time.time() - start_time)/60
print("Running time is:{:.2f} minutes.".format(time))

# Check the inertia values.

for i in range(len(ssd)):
    print('{0}: {1:.2f}'.format(i+2, ssd[i]))   

# Draw the plot to find the elbow    

plt.plot(range(2,20), ssd)
plt.xticks(range(1,20))
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.show()
 

# From the plot, we decided to choose 6 clusters for this dataset.
# Apply Models to Clusters for Preprocessed Dataset

import pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.cluster import KMeans

# Load preprocessed data

df = pd.read_csv("4-df_preprocessed.csv")
df.shape

# Sample the dataset

df_sample = df.sample(frac = 0.1,random_state = 1)
df_sample.shape

X = df_sample.iloc[:, 1:]
y = df_sample.iloc[:,0]

# Mean Score of Random Forest and Gradient Boosting for all sample data using KFold

import time
start_time = time.time()
rfm = RandomForestClassifier(n_estimators=100)
gbm = GradientBoostingClassifier(max_depth=3,n_estimators=300,learning_rate=.6)
rfm_mean_score0 = np.mean(cross_val_score(rfm,X,y,cv=5))
gbm_mean_score0 = np.mean(cross_val_score(gbm,X,y,cv=5))
time = (time.time() - start_time)/60
print("Running time is:{:.2f} minutes.".format(time))
print('Mean Score for Random Forest: {:.4f}'.format(rfm_mean_score0))
print('Mean Score for Gradient Boosting: {:.4f}'.format(gbm_mean_score0))


# Clustering the dataset based using K mean-choose 6 clusters based on the elbow method.

df_n = minmax_scale(X)
km = KMeans(n_clusters=6, random_state=1234)
# Create clusters
km.fit(df_n)

# Add the cluster number to the original data.

df_sample['ClusterNo'] = km.labels_
df_sample.head()

# Divide the original data into the clusters.

Cluster0 = df_sample.loc[df_sample.ClusterNo == 0]
Cluster1 = df_sample.loc[df_sample.ClusterNo == 1]
Cluster2 = df_sample.loc[df_sample.ClusterNo == 2]
Cluster3 = df_sample.loc[df_sample.ClusterNo == 3]
Cluster4 = df_sample.loc[df_sample.ClusterNo == 4]
Cluster5 = df_sample.loc[df_sample.ClusterNo == 5]

# Get the performance score of random forest and gradient boosting using K Fold for each cluster, get the weight for each cluster.

import time
start_time = time.time()
cluster_list = [Cluster0,Cluster1,Cluster2,Cluster3,Cluster4,Cluster5]
cluster_score_random_forest = [] # create a list to store random forest performance score
cluster_score_gradient_boosting = [] # create a list to store gradient boosting performance score
weight_list = [] # create a list to store the weight of each cluster

# Using for loop to calculate the mean performance scores for each cluster

for i in range(6):
    X = cluster_list[i].iloc[:, 1:]
    y = cluster_list[i].iloc[:,0]
    weight = round(len(y)/len(df_sample),2)
    rfm = RandomForestClassifier(n_estimators=100)
    gbm = GradientBoostingClassifier(max_depth=3,n_estimators=300,learning_rate=.6)
    rfm_mean_score = np.mean(cross_val_score(rfm,X,y,cv=5))
    gbm_mean_score = np.mean(cross_val_score(gbm,X,y,cv=5))
    cluster_score_random_forest.append(rfm_mean_score)
    cluster_score_gradient_boosting.append(gbm_mean_score)
    weight_list.append(weight)
    print('Cluster{0}-Mean Score for Random Forest: {1:.4f}'.format(i,rfm_mean_score))
    print('Cluster{0}-Mean Score for Gradient Boosting: {1:.4f}'.format(i,gbm_mean_score))
time = (time.time() - start_time)/60
print("Running time is:{:.2f} minutes.".format(time))


# Calculate weighted performance scores and compare the scores with the full sample dataset performance

cluster_score_best = list(map(max, zip(cluster_score_random_forest, cluster_score_gradient_boosting))) # best score for each cluster
weighted_score_rf = sum([cluster_score_random_forest[i] * weight_list[i] for i in range(6)]) # random forest weighted score
weighted_score_gb = sum([cluster_score_gradient_boosting[i] * weight_list[i] for i in range(6)]) # gradient boosting weighted score 
weighted_score_best = sum([cluster_score_best[i] * weight_list[i] for i in range(6)]) # weighted score by choosing best model for each cluster
print('Mean Score for Random Forest: {:.4f}'.format(rfm_mean_score0))
print('Mean Score for Gradient Boosting: {:.4f}'.format(gbm_mean_score0))
print('Weighted Score for Random Forest: {:.4f}'.format(weighted_score_rf))
print('Weighted Score for Gradient Boosting: {:.4f}'.format(weighted_score_gb))
print('Weighted Score for Best Model: {:.4f}'.format(weighted_score_best))

