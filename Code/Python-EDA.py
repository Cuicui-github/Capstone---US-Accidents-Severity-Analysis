# Read joined original data
import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv("Accidents_after_SQL.csv",sep=',', header=0)
print(df.shape)

# Statistical summary
pd.options.display.max_columns = df.shape[1]
print(df.describe(include='all'))

# Map to show accident times in different states
import folium
m = folium.Map(location=[38, -97], zoom_start=4)
url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
state_geo = f'{url}/us-states.json'
df_state = df.groupby("State").count()['Severity'].reset_index()
folium.Choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=df_state,
    columns=['State', 'Severity'],
    key_on='feature.id',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Accident times'
).add_to(m)

folium.LayerControl().add_to(m)
m

# Map to show accidents with each severity level in every state
m = folium.Map(location=[38, -97], zoom_start=4)
df_sample = df.sample(n=20000)
#print(df_sample)
df_severity_1 = df_sample[df_sample["Severity"] ==1]
df_severity_2 = df_sample[df_sample["Severity"] ==2]
df_severity_3 = df_sample[df_sample["Severity"] ==3]
df_severity_4 = df_sample[df_sample["Severity"] ==4]
group0 = folium.FeatureGroup(name='<span style=\\"color: green;\\">green circles - Severity 1</span>')
for i in range(0,len(df_severity_1)):
   folium.Circle(
      location=[df_severity_1.iloc[i]['Start_Lat'], df_severity_1.iloc[i]['Start_Lng']],
      radius=500,
      color='green',
      fill=True,
      fill_color='green'
   ).add_to(group0)
group0.add_to(m)

group1 = folium.FeatureGroup(name='<span style=\\"color: blue;\\">blue circles - Severity 2</span>')
for i in range(0,len(df_severity_2)):
   folium.Circle(
      location=[df_severity_2.iloc[i]['Start_Lat'], df_severity_2.iloc[i]['Start_Lng']],
      radius=500,
      color='#3186cc',
      fill=True,
      fill_color='#3186cc'
   ).add_to(group1)
group1.add_to(m)

group2 = folium.FeatureGroup(name='<span style=\\"color: yellow;\\">yellow circles - Severity 3</span>')
for i in range(0,len(df_severity_3)):
   folium.Circle(
      location=[df_severity_3.iloc[i]['Start_Lat'], df_severity_3.iloc[i]['Start_Lng']],
      radius=500,
      color='yellow',
      fill=True,
      fill_color='yellow'
   ).add_to(group2)
group2.add_to(m)

group3 = folium.FeatureGroup(name='<span style=\\"color: red;\\">red circles - Severity 4</span>')
for i in range(0,len(df_severity_4)):
   folium.Circle(
      location=[df_severity_4.iloc[i]['Start_Lat'], df_severity_4.iloc[i]['Start_Lng']],
      radius=500,
      color='red',
      fill=True,
      fill_color='red'
   ).add_to(group3)
group3.add_to(m)

folium.map.LayerControl('topright', collapsed=False).add_to(m)
m.save('mymap.html')

# Histogram for numerical variables

# Drop Start_Lat, Start_Lng
df = df.drop(['Start_Lat'], axis=1) 
df = df.drop(['Start_Lng'], axis=1)
df[["Is_Holiday","Lanes_Blocked","Shoulder_Blocked","Slow_Traffic","Weather_Clear", "Weather_Overcast","Weather_Dusty",     "Weather_Icy","Weather_Stormy","Bump","Crossing","Give_Way","Junction","No_Exit","Railway","Roundabout","Station",    "Stop","Traffic_Calming","Traffic_Signal"]] *= 1
df.select_dtypes(include=['int','int64','float64']).hist(bins=30, figsize=(25, 20))

#  Countplot for single variables
sns.countplot(x = "Severity", data = df)
sns.countplot(x = "Month", data = df[df["Year"] == 2018]) 
sns.countplot(x = "Daylight", data = df)
df_state_top = df.groupby("State").count()['Severity'].reset_index().sort_values("Severity",ascending=False).head(10)
df_state_top.columns = ['State', 'Accidents']
sns.barplot(df_state_top["State"], df_state_top["Accidents"])
sns.countplot(x = "Street_Type", data = df)
sns.countplot(x = "Weekday", data = df, order = df['Weekday'].value_counts().index)
sns.countplot(x = "Hour", data = df)

# Generate the import plots used in the main text
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,2,figsize=(15, 10))
sns.barplot(df_state_top["State"], df_state_top["Accidents"], ax = ax[0,0])
sns.countplot(x = "Street_Type", data = df,ax = ax[0,1])
sns.countplot(x = "Weekday", data = df, order = df['Weekday'].value_counts().index,  ax = ax[1,0])
ax[1,0].set(xlabel = 'Day of Week') 
sns.countplot(x = "Hour", data = df, ax = ax[1,1])
 

# Plots for different severity
sns.countplot(x="Month", hue="Severity", data=df[df["Year"] == 2018])#using 2018 as example
sns.countplot(x="Hour", hue="Severity", data=df)
sns.countplot(x="Daylight", hue="Severity", data=df)
ax = sns.countplot(x="Weekday", hue="Severity", data=df,order = df['Weekday'].value_counts().index )
ax.set(xlabel = 'Day of Week')
df["Weekend"] = 0
df.loc[(df['Weekday']== "Saturday") |(df['Weekday']== "Sunday"),"Weekend"] = 1
sns.countplot(x="Weekend", hue="Severity", data=df)
sns.countplot(x="Street_Type", hue="Severity", data=df)
df["Freeway"] = 0
df.loc[(df['Street_Type']== "Freeway"), "Freeway"] = 1
sns.countplot(x="Freeway", hue="Severity", data=df)

#  Generate the import plots used in the main text
fig, ax = plt.subplots(2,2,figsize=(15, 10))
sns.countplot(x="Street_Type", hue="Severity", data=df, ax = ax[0,0])
sns.countplot(x="Freeway", hue="Severity", data=df,ax = ax[0,1])
sns.countplot(x="Weekday", hue="Severity", data=df,order = df['Weekday'].value_counts().index, ax = ax[1,0])
ax[1,0].set(xlabel = 'Day of Week') 
sns.countplot(x="Weekend", hue="Severity", data=df, ax = ax[1,1])
