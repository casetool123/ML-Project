#!/usr/bin/env python
# coding: utf-8

# ## price predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS']


# In[6]:


housing['CHAS'].value_counts()


# In[7]:


housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))


# ## train test splitting
# 

# In[10]:


#For knowing purpose what is actually happening in cell 13.. 
#train_test_split function is already present in sklearn as used below
import numpy as np

def split_train_test(data, test_ratio):
    np.random.seed(42) 
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

    


# In[11]:


#train_set, test_set = split_train_test(housing, 0.2)


# In[12]:


#print(f"Rows in train_set: {len(train_set)}\nRows in test_set: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
print(f"Rows in train_set: {len(train_set)}\nRows in test_set: {len(test_set)}\n")


# In[14]:


#we use stratified sampling to represent all the variables equally in test set and train set...
#we are applying in CHAS here to get equal ratio of 0 and 1 in train and test set
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[15]:


strat_test_set.info()


# In[16]:


strat_test_set['CHAS'].value_counts()


# In[17]:


strat_train_set.info()


# In[18]:


strat_train_set['CHAS'].value_counts()


# In[19]:


95/7


# In[20]:


376/28
#we can see that we almost got the same ratio of 0 and 1 in train and test set..
#this is the work of stratified sampling.


# In[21]:


housing = strat_train_set.copy() #housing data get copied before we find coorelation


# ## looking for coorelation

# In[22]:


#corr function present in pandas
corr_matrix = housing.corr()


# In[23]:


#as MEDV is the median value of price.. we will find correlation wrt MEDV 
corr_matrix['MEDV'].sort_values(ascending = False)


# In[24]:


#now we will plot the above variables which have strong coorelations
from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[25]:


housing.plot(kind = "scatter", x = "RM", y = "MEDV", alpha=0.8)


# In[26]:


#combining attributes to make a new attribute
housing["TAXRM"] = housing["TAX"] / housing["RM"]


# In[27]:


housing["TAXRM"]


# In[28]:


housing.head()


# In[29]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[30]:


housing.plot(kind = "scatter", x = "TAXRM", y = "MEDV", alpha=0.8)


# In[31]:


housing = strat_train_set.drop("MEDV", axis = 1)
housing_labels = strat_train_set["MEDV"].copy()


# ##  missing values will get filled my median

# In[32]:


#working with only train set
housing.shape


# In[33]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[34]:


imputer.statistics_   #show the calculated median values


# In[35]:


#automatically the missing values gets filled
X = imputer.transform(housing)


# In[36]:


housing_tr = pd.DataFrame(X, columns= housing.columns)


# In[37]:


housing_tr.describe()


# ## creating a pipeline(feature scalling)

# In[38]:


from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('std_scalar', StandardScaler()),
])


# In[39]:


housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr


# In[40]:


housing_num_tr.shape


# ## selecting desire model

# In[41]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[42]:


some_data = housing.iloc[:5]


# In[43]:


some_labels = housing_labels.iloc[:5]


# In[44]:


prepared_data = my_pipeline.transform(some_data)


# In[45]:


model.predict(prepared_data)


# In[46]:


list(some_labels)


# ## evaluating the model

# In[47]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[48]:


mse


# In[49]:


rmse


# ## using better validation techniques- cross validation

# In[50]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)


# In[51]:


rmse_scores


# In[52]:


def print_scores(scores):
    print("scores: ", scores)
    print("mean: ", scores.mean())
    print("standard deviation: ", scores.std())


# In[53]:


print_scores(rmse_scores)


# ## saving model

# In[54]:


from joblib import dump, load
dump(model, 'price.joblib')


# ## testing the model

# In[55]:


#now we will work with the test set
X_test = strat_test_set.drop("MEDV", axis = 1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[56]:


final_rmse


# In[57]:


prepared_data[0]


# ## using the model

# In[58]:


from joblib import dump, load
import numpy as np
model = load('price.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])

model.predict(features)


# In[ ]:




