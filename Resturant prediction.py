import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filter('ignore')

# import dataset 


# dataset import
# load dataset
df = pd.read_csv('Dataset.csv')
df.head()

# Check shape
df.shape

# Dataset information
df.info

# Missing values
df.isnull().sum()

# Handling Missing Values
df.dropna(inplace=True)

# Data type conversion
df['Average Cost for two'] = df['Avearge Cost for two'].astype(float)

# Aggregate rate
sns.histplot(df['Aggregate rating'],bins=10)
plt.show()

#=====================
#LEVEL 1 - TASK 2
#=====================

# Descriptive Analysis
df.describe()

# categorical analysis
df['City'].value_counts().head(15)
df['Cuisines'].value_counts().head(15)

# countplot
sns.countplot(y=df['City'],order=df['City'].value_counts().index[:10])
plt.show()

# countplot (2)
sns.countplot(y=df['Cuisines'],order=df['Cuisines'].value_counts().index[:10])
plt.show()

#geospatial analysis

import folium

# create map
m = folium.Map(
    location=[df['Latitude'].mean(), df['Longitude'].mean()],
    zoom_start=5
)

# Add markers 
for i in range(len(df)):
    folium.Marker(
        [df.iloc[i]['Latitude'], df.iloc[i]['Longitude']]
    ).add_to(m)

    # Display map
    m

    # saving map
    m.save("restuarants_map.html")

    #======================
    # LEVEL 2 - TASK 1
    #======================

    # percentage of restuarants with table booking
    df['Has Table booking'].value_counts(normalize=True)

    # compare ratings
    df.groupby('Has Table booking')['Aggregate rating'].mean()

    # bar graph
    sns.barplot(x='Has Table booking', y='Aggregate rating', data=df)
    plt.show()

    # Task 2 

    df['Price range'].value_counts()
    df.groupby('Price range')['Aggregate rating'].mean()

    # pie chart
    df['Price range'].value_counts().plot.pie(
        autopct='%1.1f%%',
        labels=None
    )

    plt.ylabel('')
    plt.show()

    #======================
    #LEVEL 2 - TASK 3
    #======================

    df['Name_Length'] = df['Restaurant Name'].apply(len)

    df[['Restaurant Name', 'Name_Length']].head()

    # ENCODING 

    from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Has Online delivery'] = le.fit_transform(df['Has Online delivery'])

# UNIQUE

print(df['Has Online delivery'].unique()) 

#==========================
# LEVEL 3 - TASK 1
#==========================

# Train & Test split

from sklearn.model_selection import train_test_split
x = df.drop('Aggregate rating', axis=1)
y = df['Aggregate rating']

print(x.head())
print(y.head())

# Check string columns

print(x.select_dtypes(include='object').columns)

x = pd.get_dummies(x, drop_first=True)

# Linear Regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x, y)

y_pred = lr.predict(x)

print(y_pred)

# Decision Tree Regressor 

from sklearn.tree import DecisionTreeRegressor # Changed to Regressor
dt = DecisionTreeRegressor() # Changed to Regressor
dt.fit(x, y)

y_pred = dt.predict(x)

print(y_pred)

# Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor # Changed to Regressor
rf = RandomForestRegressor() # Changed to Regressor
rf.fit(x, y)

y_pred = rf.predict(x)

print(y_pred)

# Level 3 - Task 2

df.groupby('Cuisines')['Votes'].sum().sort_values(ascending=False).head(10)

# Level 3 - Task 3

# Histogram

import matplotlib.pyplot as plt
import seaborn as sns
# Histogram
sns.histplot(df['Aggregate rating'])
plt.show()

# Box plot

import matplotlib.pyplot as plt
import seaborn as sns
# Box plot
sns.boxplot(x=df['Aggregate rating'])
plt.show()

# Distplot

sns.displot(df['Aggregate rating'])
plt.show()
































    




