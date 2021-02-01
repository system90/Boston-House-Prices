#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Sklearn Linear regression on Boston House dataset


# In[2]:


# import libraries:
import numpy as np
import pandas as pd

# Visualization Libraries:
import seaborn as sns
import matplotlib.pyplot as plt

# To plot the graph embedded in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Imports sklearn modules:

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


# In[4]:


# Load the dataset and get the dataframe dimensions:

boston = datasets.load_boston()
print(boston.data.shape)


# In[5]:


# Print the dataset:

print(boston)


# In[6]:


# Print information about dataset:

print(boston.DESCR)


# In[7]:


# Apply dataframe using Pandas:

bos = pd.DataFrame(boston.data, columns=boston.feature_names)
print(bos)


# There are 13 features/columns in this dataset

# In[8]:


# Add target variable (extra column) to dataset. The PRICE data is the median value of owener occupied homes ($1000s)

bos['PRICE'] = boston.target
print(bos)


# The dataset now has 14 features/columns. New column: 'PRICE'

# In[9]:


# Check the dataset for any missing values:

bos.isnull().sum()


# There is no missing data

# In[10]:


# Obtain the mean, median, standard deviation etc of each feature/column:
print(bos.describe())


# ## Exploratory data analysis
# 
# Observe the difference between the target variable and other features.

# In[11]:


# Plot distribution of target variable

#sns.set(rc={'figure.figsize':(11.7,8.27)})

plt.hist(bos['PRICE'], bins=30)
plt.xlabel("House prices in $1000")
plt.ylabel("Count")
plt.title("House Price Histogram", fontsize=25)


# The histogram shows that the data normally distributed, with some outliers.

# In[26]:


# Correlation matrix (heatmap) to measure the correlation between all of the variables.

bos_1 = pd.DataFrame(boston.data, columns = boston.feature_names) # create a dataframe with the 13 original features (without the PRICE column)


corr_mat = bos_1.corr().round(2)  # correlation of bos_1 dataframe, (round to 2 dp)
print(corr_mat)


# Above we can see the correlation values between each of the varaibles. Now a heatmap can be created based on these values.

# In[54]:


# Create the heatmap:

plt.figure(figsize = (11,8))                              # Plotsize
sns.heatmap(data=corr_mat, annot=True, cmap='coolwarm')   # Heatmap with numerical values  


# The red boxes represents a stronger negative correlation. Whilst the blue boxes reprents a stonger positive correlation.
# There is a strong negative correlation between NOX and DIS (-0.77). Whilst there is a strong positive correlation between RAD and TAX (0.91).

# In[53]:


# Split data into INDEPENDENT variables and TARGET variable:

x = bos.drop(['PRICE'], axis = 1)    # INDEPENDENT data
y = bos['PRICE']                     # TARGET data
print(y)


# In[55]:


# Split data into TRAIN and TEST. Allocate 30% of data to test set

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=4)


# In[58]:


# Check the sizes of the datasets
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ## Linear Regression

# In[61]:


# Train the model

lm = LinearRegression()    # create the linear regressor
lm.fit(x_train, y_train)   # Train the model using the 2 training sets


# In[62]:


# Value of the y-intercept

lm.intercept_

