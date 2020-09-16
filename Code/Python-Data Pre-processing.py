# Handling Null Values
#Check the data size and columns data type
import pandas as pd
import numpy as np
df = pd.read_csv("Accidents_after_SQL.csv",sep=',', header=0)
print(df.shape)
print(df.info())

# Change data type as needed, show the statistic summary of dataset
df[["Lanes_Blocked","Crossing","Junction","Railway","Station","Stop","Traffic_Signal"]] = df[["Lanes_Blocked","Crossing",    "Junction","Railway","Station","Stop","Traffic_Signal"]].astype(bool)
pd.options.display.max_columns = df.shape[1]
print(df.describe(include='all'))

# 'Turning_Loop' is a constant column, need to drop. Check the null ratio for the remaining columns
df = df.drop(['Turning_Loop'], axis=1)
print(df.isnull().sum()/len(df))

# Drop null values since the ratio is very low, show the summary of data after drop the null values.
df1 = df.dropna()
print(df1.shape)
print(df1.head())
pd.options.display.max_columns = df1.shape[1]
print(df1.describe(include='all'))

# The data providers mentioned they changed the method of collecting data from Aug, 2017. Some states in the following were impacted. 
state_list = ["AL","AR","AZ","CO","ID","KS","KY","LA","ME","MN","MS","MT","NC","ND","NH","NM", "NV","OK","OR","SD","TN","UT","VT","WI","WY"]
df1.loc[df1['State'].isin(state_list)].groupby([df1["Year"], df1["Month"]]).count()["Severity"].plot(kind="bar")

# After checking the distribution, we decided to drop the records before Aug, 2017 to keep all the records collected in the same way.
df1 = df1.loc[(df1["Year"]>2017) | ((df1["Year"]==2017) & (df1["Month"]>7))]

# Double check after remove the records
df1.groupby([df1["Year"], df1["Month"]]).count()["Severity"].plot(kind="bar")

# Show summary of data after the cleaning
print(df1.shape)
pd.options.display.max_columns = df1.shape[1]
print(df1.describe(include='all'))

# Save null cleaned data into new csv
df1.to_csv("1-df_null_cleaned.csv", index = False)

#Outliers Analysis

# Read null cleaned data
import pandas as pd
import numpy as np
from matplotlib.pyplot import boxplot, hist
df = pd.read_csv("1-df_null_cleaned.csv")
print(df.shape)

# Outlier Analysis
# This function takes a pandas Series and returns a tuple of
# upper and lower boundary using the boxplot.

def whiskers (c):
    c_desc = c.describe()
    q1 = c_desc['25%']
    q3 = c_desc['75%']
    iqr = q3 - q1 #interquartile range
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    return{'upper': upper,'lower': lower}

# This function takes a DataFrame and returns a DataFrame containing
# the number of outliers above and below the upper and lower boundary, 
# respectively for each column.

def num_outliers(df):
    cols = ['ColName','Upper outliers','Lower outliers','Outlier Ratio']
    outlier_table = pd.DataFrame(columns=cols)
        
    for c in df.columns:
        s = df[c]
        n_upper = s[s>whiskers(s)['upper']].count()
        n_lower = s[s < whiskers(s)['lower']].count()
        ratio = (n_upper + n_lower)/len(df[c])
        outlier_row = pd.DataFrame([[c,n_upper,n_lower,ratio]],columns=cols)
        outlier_table = pd.concat([outlier_table, outlier_row], axis=0,ignore_index=True)
    return outlier_table

# This takes two DataFrames and returns a DataFrame after removing outliers.
# df1 only contains the numerical variables, df is the full dataframe

def delete_outliers(df1, df):
    for c in df1.columns:
        s = df[c]
        if s[s>whiskers(s)['upper']].count() > 0:
            df = df.drop (df[df[c]> whiskers(s)['upper']].index)
        if s[s>whiskers(s)['lower']].count() > 0:
            df = df.drop (df[df[c] < whiskers(s)['lower']].index)
    return df

# Check outliers for numerical variables
df_outlier = df
num_outliers(df_outlier.select_dtypes(include=['int','int64','float64']))
Table 1. Outlier Summary was generated from this output
		 

# Analyze and deal with the outliers for each column
import seaborn as sns

# Duration_minutes
sns.distplot(df_outlier["Duration_minutes"])
sns.boxplot(df_outlier["Duration_minutes"])
print(len(df_outlier[df_outlier["Duration_minutes"]<=0]))
print(len(df_outlier[df_outlier["Duration_minutes"]<0]))

# We need to drop the records with "Duration_minutes" <= 0
df_outlier = df_outlier[df_outlier["Duration_minutes"]>0]

# We removed the negative values for "Duration_minutes" because negative values don’t make sense for duration minutes.

# Distance_mi
df_outlier = df.drop(['Distance_mi'], axis=1)

# For "Distance_mi", it is one result of an accident instead of a causal variable, we decided to drop this column.

# Temperature_F
sns.distplot(df_outlier["Temperature_F"])
sns.boxplot(df_outlier["Temperature_F"])

# Delete outliers for "Temperature_F"
df_outlier = delete_outliers(df_outlier[["Temperature_F"]],df_outlier)

# Since there were not so many outliers for "Temperature_F", we decided to remove the extreme values for it.

# Pressure_in
sns.distplot(df_outlier["Pressure_in"])
sns.boxplot(df_outlier["Pressure_in"])

# The barometric pressure seldom goes above 31 inches or drops below 29 inches.
whiskers (df_outlier["Pressure_in"])

# Based on the research, we decided to keep the "Pressure_in" in the range of 29 ~ 31
df_outlier = df_outlier[(df_outlier["Pressure_in"] >= 29) & (df_outlier["Pressure_in"] <= 31)]

# Based on the research, we decided to keep the "Pressure_in" in the range of 29 ~ 31.  
#https://www.infoplease.com/math-science/weather/weather-is-the-pressure-getting-to-you#:~:text=The%20weight%20of%20the%20atmosphere,level%20pressure%20is%2029.92%20inches.

# Visibility_mi
sns.distplot(df_outlier["Visibility_mi"])
sns.boxplot(df_outlier["Visibility_mi"])
print(len(df_outlier[df_outlier['Visibility_mi']>=20])/len(df_outlier))
# Remove the values over 20
df_outlier = df_outlier[df_outlier['Visibility_mi']<20]

# Based on the distribution, we decided to drop the values over 20 for "Visibility_mi". 

# Precipitation_in
sns.distplot(df_outlier["Precipitation_in"])
sns.boxplot(df_outlier["Precipitation_in"])
print(len(df_outlier[df_outlier['Precipitation_in']== 0])/len(df_outlier))
print(len(df_outlier[df_outlier['Precipitation_in']<0]))
print(len(df_outlier[df_outlier['Precipitation_in']==0]))
print(len(df_outlier[df_outlier['Precipitation_in']!=0]))

		# Group Precipotation_in into zero or not-zero
df_outlier["Precipitation"] = 0
df_outlier.loc[df_outlier['Precipitation_in']!= 0,"Precipitation"] = 1
print(df_outlier["Precipitation"].value_counts())

# "Precipitation_in" contains 92.82% zeroes, we decided to group it into zero or not-zero.

# Wind_Speed_mph
sns.distplot(df_outlier["Wind_Speed_mph"])
sns.boxplot(df_outlier["Wind_Speed_mph"])
print(len(df_outlier[df_outlier['Wind_Speed_mph']>30]))
print(len(df_outlier[df_outlier['Wind_Speed_mph']>30])/len(df_outlier))

# Remove values above 30
df_outlier = df_outlier[df_outlier['Wind_Speed_mph']<=30]
sns.distplot(df_outlier["Wind_Speed_mph"])
sns.boxplot(df_outlier["Wind_Speed_mph"])

# Based on the research, winds of 30 to 45 mph can make driving significantly dangerous. We decided to remove the values above 30.  
#https://bartbernard.com/truck-accidents/five-tips-for-driving-in-gusty-winds/#:~:text=Winds%20of%20even%2030%20to,other%20motorists%20on%20the%20road.

# Pop_Density_pplPerMeter
sns.distplot(df_outlier["Pop_Density_pplPerMeter"])
sns.boxplot(df_outlier["Pop_Density_pplPerMeter"])
whiskers(df_outlier["Pop_Density_pplPerMeter"])
len(df_outlier[df_outlier["Pop_Density_pplPerMeter"]<0])
df_outlier = delete_outliers(df_outlier[["Pop_Density_pplPerMeter"]], df_outlier)

fig, (ax1, ax2) = plt.subplots(ncols=2)
sns.distplot(df_outlier["Pop_Density_pplPerMeter"], ax = ax1)
sns.boxplot(df_outlier["Pop_Density_pplPerMeter"], ax = ax2)

# We dropped the outliers for Pop_Density_pplPerMeter because there may be some errors in the data and the proportion of outliers is not high.

# Summary after dealing with outliers
print(df_outlier.shape)
pd.options.display.max_columns = df_outlier.shape[1]
print(df_outlier.describe(include='all'))

# Save data to new csv
df_outlier.to_csv("2-df_outlier.csv", index = False)

#Correlation Analysis

# Read data after dealing with outliers
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv("2-df_outlier.csv")
print(df.shape)

# Statistical summary
pd.options.display.max_columns = df.shape[1]
print(df.describe(include='all'))

# Drop Start_Lat, Start_Lng
df = df.drop(['Start_Lat'], axis=1) 
df = df.drop(['Start_Lng'], axis=1)

# Change bool variables to 0 and 1
df[["Is_Holiday","Lanes_Blocked","Shoulder_Blocked","Slow_Traffic","Weather_Clear", "Weather_Overcast","Weather_Dusty",     "Weather_Icy","Weather_Stormy","Bump","Crossing","Give_Way","Junction","No_Exit","Railway","Roundabout","Station",    "Stop","Traffic_Calming","Traffic_Signal"]] *= 1

# Plots with different severities
sns.countplot(x="Weekday", hue="Severity", data=df,order = df['Weekday'].value_counts().index )

# Weekends has a higher rate of high severity, we can group into weekend or not
df["Weekend"] = 0
df.loc[(df['Weekday']== "Saturday") |(df['Weekday']== "Sunday"),"Weekend"] = 1
sns.countplot(x="Weekend", hue="Severity", data=df)

# Drop useless columns
df = df.drop(['Year'], axis=1) 
df = df.drop(['State'], axis=1) 
df = df.drop(['City'], axis=1) 
df = df.drop(['County'], axis=1)

# Correlation Test for Numerical Data
plt.figure(figsize = (20,15))
sns.heatmap(df.select_dtypes(include=['int','int64','float64']).corr(), annot=True,fmt='.2f', linewidths=.5,cmap="PiYG")

# Show Pearson correlation for numerical variables
from scipy import stats
df1 = df.select_dtypes(include=['int','int64','float64'])
corr_table = pd.DataFrame(columns=['col1','col2','Pearson_correlation','p-value'])
for i in range(len(df1.columns)-1):
    for j in range(len(df1.columns)-1-i):
        if abs(stats.pearsonr(df1.iloc[:,i],df1.iloc[:,i+j+1])[0]) >= 0.10:
            corr_table = corr_table.append({'col1':df1.columns[i],'col2':df1.columns[i+j+1],'Pearson_correlation':stats.pearsonr(df1.iloc[:,i],df1.iloc[:,i+j+1])[0],                                            'p-value':stats.pearsonr(df1.iloc[:,i],df1.iloc[:,i+j+1])[1]},ignore_index=True)
        else:
            continue
print(corr_table.reindex(corr_table.Pearson_correlation.abs().sort_values(ascending = False).index))
		 

# Chi-square test for nominal data
# The chi-square test shows only whether we can reject or fail to reject Ho.
# For the level of correlation, we need to calculate Cramer's V (or phi) to 
# measure the association between two nominal variables.  
# Cramer's V ranges between 0 and 1 (completely correlated).

def cramer_v(x, y):
    n = len(x)
    ct = pd.crosstab(x, y) # crosstab
    chi2 = stats.chi2_contingency(ct)[0]
    v = np.sqrt(chi2 / (n * (np.min(ct.shape) - 1)))
    return v
chi_table = pd.DataFrame(columns = ["col1","col2","Cramers_V","p-value"])
cols = [col for col in df.columns if col not in ["Duration_minutes","Temperature_F","Pressure_in","Visibility_mi","Precipitation_in","Wind_Speed_mph","Pop_Density_pplPerMeter"]]
df2 = df[cols]
for i in range(len(df2.columns)-1):
    for j in range(len(df2.columns)-1-i):
        if abs(cramer_v(df2.iloc[:,i],df2.iloc[:,i+j+1])) >= 0.10:
            chi_table = chi_table.append({'col1':df2.columns[i],'col2':df2.columns[i+j+1],                                         "Cramers_V":cramer_v(df2.iloc[:,i],df2.iloc[:,i+j+1]),                                         'p-value':stats.chi2_contingency(pd.crosstab(df2.iloc[:,i],df2.iloc[:,i+j+1]))[1]},ignore_index=True)
        else:
            continue
print(chi_table.reindex(chi_table.Cramers_V.abs().sort_values(ascending = False).index))
 
# Based in the correlation analysis, we decided to remove Weekday, Duration_minutes, Hour
df1 = df.drop(["Weekday","Hour","Duration_minutes"],axis=1)

# save new data
print(df1.shape)
df1.to_csv("3-df_EDA.csv", index = False)

#Converting nominal values to numerical values
#Load dataset
import pandas as pd
import numpy as np
df = pd.read_csv("3-df_EDA.csv")
print(df.shape)
print(df.info())

# Get dummy variables for categorical variables
df1 = pd.get_dummies(df,columns=['Daylight'])
df1 = pd.get_dummies(df1,columns=['Street_Type'])
df1.info()

# Find the correlation for each column of dummy variables
from scipy import stats
def cramer_v(x, y):
    n = len(x)
    ct = pd.crosstab(x, y) # crosstab
    chi2 = stats.chi2_contingency(ct)[0]
    v = np.sqrt(chi2 / (n * (np.min(ct.shape) - 1)))
    return v
print('Cramers V between Severity and Daylight_Day: ', {cramer_v(df1["Severity"],df1["Daylight_Day"])})
print('Cramers V between Severity and Daylight_Night: ', {cramer_v(df1["Severity"],df1["Daylight_Night"])})
print('Cramers V between Severity and Daylight_Twilight: ', {cramer_v(df1["Severity"],df1["Daylight_Twilight"])})
print('Cramers V between Severity and Street_Type_Boulevard: ', {cramer_v(df1["Severity"],df1["Street_Type_Boulevard"])})
print('Cramers V between Severity and Street_Type_Freeway: ', {cramer_v(df1["Severity"],df1["Street_Type_Freeway"])})
print('Cramers V between Severity and Street_Type_Highway: ', {cramer_v(df1["Severity"],df1["Street_Type_Highway"])})
print('Cramers V between Severity and Street_Type_Road: ', {cramer_v(df1["Severity"],df1["Street_Type_Road"])})
print('Cramers V between Severity and Street_Type_Street/Ave: ', {cramer_v(df1["Severity"],df1["Street_Type_Street/Ave"])})
print('Cramers V between Severity and Street_Type_dead_end: ', {cramer_v(df1["Severity"],df1["Street_Type_dead_end"])})

# Drop the lowest correlation dummy variable.
df1.drop(["Daylight_Twilight","Street_Type_dead_end"], inplace = True, axis = 1)
# Drop duplicated column
df1.drop(["Precipitation_in"], inplace = True, axis = 1)

# save new data
print(df1.shape)
df1.to_csv("4-df_preprocessed.csv", index = False)

#Primary Component Analysis: 

# Load the dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
df3 = pd.read_csv("1-df_null_cleaned.csv")
print(df3.shape)
print(df3.info())
df3 = pd.get_dummies(df3,columns=['Daylight'])
df3 = pd.get_dummies(df3,columns=['Street_Type'])
df3["Weekend"] = 0
df3.loc[(df3['Weekday']== "Saturday") |(df3['Weekday']== "Sunday"),"Weekend"] = 1
df3.drop(["Daylight_Twilight","Street_Type_dead_end","Weekday"],inplace = True, axis = 1)
df3 = df3.loc[:, df3.dtypes != "object"]
df3.iloc[:,[3,9,10,11,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]] *= 1
df3.drop(df3.columns[[1,5,6,7,8]], axis=1, inplace=True)

# Divide the data into X and y
y = df3.iloc[:,0]
X = df3.iloc[:,1:]
# Handling null values
# Standardize the X
zN = StandardScaler()
dfn3 = zN.fit_transform(X)

# Draw a scree plot and determine the number of PCA components
pca3 = PCA().fit(dfn3)
fig, axs = plt.subplots(2,figsize=(10, 10))
axs[0].plot(pca3.explained_variance_ratio_)
axs[0].set_title('explained variance')
axs[1].plot(np.cumsum(pca3.explained_variance_ratio_))
axs[1].set_title('cumulative explained variance')

  
pca3.explained_variance_ratio_
np.cumsum(pca3.explained_variance_ratio_)
 

# Save 95% variance ratio dataset for later model analysis
# when the cumulative is at least 95%, the least components are 31
pca = PCA (n_components=31, random_state=1234)
# Xp has now 31 columns of primary components.
Xp = pca.fit_transform(dfn3)
Xp = np.round(Xp,2)
df_PCA = np.column_stack((y.to_numpy(), Xp))
# save data
print(df_PCA.shape)
pd.DataFrame(df_PCA).to_csv("5-df_PCA_1.csv", index = False)
