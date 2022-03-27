#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
import matplotlib.ticker as mtick
#plt.style.use("fivethirtyeight")
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# In[2]:


# Just for disabling warning masseges
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Importing zomato file in csv format
df1= pd.read_csv("C:/Users/Abhinav kumar/Desktop/zomato.csv")


# In[4]:


# checking top 10 rows
df1.head(10)


# In[5]:


# checking last 10 rows
df1.tail(10)


# ### column description
# 
# url                            url contains the link of the restaurent in zomato website
# address                        address contains address of the restaurent in Bengaluru
# name                           name contains name of the restaurent
# online_order                   whether online ordering is available in the resturent or not
# book_table                     table booking option is available or not
# rate                           overall rating of the restaurent 
# votes                          contains total no of ratings 
# phone                          contains phone number of restaurent 
# location                       contains nearby landmark location
# rest_type                      type of the restaurent
# dish_liked                     dishes liked by the customer
# cuisines                       food style separated by comma
# approx_cost(for two people)    approximate cost of the meal for two people
# reviews_list                   list of tuples containing review of the restaurent
# menu_item                      contains list of the menu available in the restaurent
# listed_in(type)	               type of the meal
# listed_in(city)                location neighbourhood in which restaurent is listed 

# In[6]:


# checking shape of the dataframe
df1.shape


# In[7]:


# checking the types of the features
df1.dtypes


# In[8]:


## GETTING FEW MORE INFORMATION THROUGH THIS 
df1.info()


# In[9]:


# checking for the total null value in any feature
df1.isnull().sum()


# In[10]:


# Group and aggregate duplicate restaurants that are listed under multiple types in listed_in(type)
#grouped = df1.groupby(["name", "address"]).agg({"listed_in(type)_x" : list})
#df = pd.merge(grouped, df1, on = (["name", "address"]))


# In[11]:


#deleting unnecessary features
df=df1.drop(['url','phone'], axis=1)


# In[12]:


#checking for duplicate rows
df.duplicated().sum()


# In[13]:


df.drop_duplicates(inplace=True)


# In[14]:


df.duplicated().sum()


# In[15]:


# Drop rows which have duplicate information in "name", "address" and "listed_in(type)_x"
#df["listed_in(type)_x"] = df["listed_in(type)_x"].astype(str) 
# converting unhashable list to a hashable type
#df.drop_duplicates(subset = ["name", "address", "listed_in(type)_x"], inplace = True)


# In[16]:


df.shape


# ### Drop null values

# In[17]:


# Remove the NaN values from the dataset
df.dropna(how='any', inplace= True)
df.isnull().sum()


# In[18]:


df.shape


# shape reduced from 51717 to 23248 dropping 28469 rows

# In[19]:


df.columns


# In[20]:


# Renaming the column name
df=df.rename(columns={'approx_cost(for two people)':'cost',
                      'listed_in(type)':'type', 'listed_in(city)':'city'})
df.columns


# In[21]:


df.head()


# ### Data cleaning

# In[22]:


# looking for all unique values in cost column
df['cost'].unique()


# In[23]:


# replacing the comma with nothing and making data in float
df['cost']= df['cost'].apply(lambda x: x.replace(",",""))


# In[24]:


# no of unique values of cost
df['cost'].unique()


# In[25]:


# Changing the attribute type to float
df['cost']= df['cost'].astype(float)


# In[26]:


df['cost'].unique()


# In[27]:


# checking for the type of the objects
df.dtypes


# In[28]:


df['rate'].unique()


# In[29]:


# removing NEW 
df=df.loc[df.rate !='NEW']


# In[30]:


df['rate'].unique()


# In[31]:


# romoving /5 that is (out of five) rating from rate
df['rate']=df['rate'].apply(lambda x: x.replace("/5",""))


# In[32]:


df['rate'].unique()


# In[33]:


#converting it into float
df['rate']=df['rate'].astype(float)


# In[34]:


# Examining restaurant types in the column "listed_in(type)"
df["type"].value_counts()


# ### Data Visualization

# In[35]:


# Most famous restaurent in Bengaluru
popularchain= df['name'].value_counts()[:20]
sns.barplot(x=popularchain, y=popularchain.index,palette='deep')
plt.title('Most popular restaurant chain in bengaluru')
plt.xlabel('No. of outlets')
plt.figure(figsize=(12,10))
plt.show()


# ### Whether restaurent offers table booking

# In[36]:


tablebooking=df['book_table'].value_counts()
sns.barplot(x=tablebooking, y=tablebooking.index,palette='deep')
plt.title('No of table booking restaurants in bengaluru')
plt.xlabel('No. of Restaurents')
plt.figure(figsize=(12,10))
plt.show()


# ### Insite
# Most of the restaurent does not provide table booking

# In[37]:


# Converting the restaurant names to rownames 
df.index = df["name"]
# Identifying the top 10 cuisines in Bangalore?
pd.DataFrame(df.groupby(["cuisines"])["cuisines"].agg(['count']).sort_values("count", ascending = False)).head(10)


# In[38]:


#whether restaurents deliver online or not
sns.countplot(df['online_order'])
fig=plt.gcf()
fig.set_size_inches(6,4)
plt.title("whether restaurents deliver online or not")
plt.show()


# ### Insite
# Many restaurent offers online orders

# ### Rating distribution

# In[39]:


# Plotting the distribution of restaurant ratings
plt.figure(figsize = (8, 6))
plt.hist(df.rate, bins = 20, color = "r")
plt.show()


# 1. Almost more than 70 percent of restaurants has rating between 3 and 4.5
# 2. Restaurants having rating more than 4.5 are very rare.

# In[40]:


# Printing restaurant value counts for the top 30 locations
df["location"].value_counts()[:30]


# In[41]:


# Top 10 locations with the highest ratings
(pd.DataFrame(df.groupby("location")["rate"].mean())).sort_values("rate", ascending = False).head(10)


# In[42]:


# Top 10 most expensive locations (cost = cost for two)
(pd.DataFrame(df.groupby("location")["cost"].mean())).sort_values("cost", ascending = False).head(10)


# ### count of restaurents from rating 1 to 2, 2 to 3, 3 to 4, 4 to 5

# In[43]:


df['rate'].unique()


# In[44]:


df['rate'].min()


# In[45]:


df['rate'].max()


# In[46]:


df['rate']=df['rate'].astype(float)


# In[47]:


((df['rate']>=1) & (df['rate']<2)).sum()


# In[48]:


((df['rate']>=2) & (df['rate']<3)).sum()


# In[49]:


((df['rate']>=3) & (df['rate']<4)).sum()


# In[50]:


#((df['rate']>=4) & (df['rate']<4.5)).sum()
# 10675


# In[51]:


((df['rate']>=4.5) & (df['rate']<5)).sum()


# In[52]:


autopct='%1.0f%%'


# ### plotting the rating counts with pie chart

# In[53]:


slices=[((df['rate']>=1) & (df['rate']<2)).sum(),
        ((df['rate']>=2) & (df['rate']<3)).sum(),
        ((df['rate']>=3) & (df['rate']<4)).sum(),
        ((df['rate']>=4) & (df['rate']<5)).sum()]
labels=['1<=rate<2','2<=rate<3','3<=rate<4','4<=rate<5']
plt.pie(slices,labels=labels, autopct='%1.0f%%',pctdistance=0.5,labeldistance=1.2,shadow=True)
fig=plt.gcf()
plt.title("Percentage of restaurent according to their ratings")
fig.set_size_inches(5,5)
plt.show()


# ### Service types

# In[54]:


sns.countplot(df['type']).set_xticklabels(sns.countplot(df['type']).get_xticklabels(),
                                           rotation=90,ha="right")
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.title("Type of services")
plt.tight_layout()


# ### Insite
# The two most frequent services are Delivery and Dine-out 

# ### Distribution of Charges(cost per two person)

# In[55]:


plt.figure(figsize=(6,6))
sns.distplot(df['cost'])
plt.show()


# In[56]:


# Visualizing the relationship between rating and cost
plt.figure(figsize = (10, 5))
plt.scatter(df.rate, df.cost)
plt.show()


# Maximum cost is in restaurents having ratings between 3.7 to 4.7. Surprisingly the highest cost is in 4.4(approx) rated restaurent and restaurent rated nearly 4.9 has far less cost

# In[57]:


# re=regular expression used for splitting words
import re
df.index=range(df.shape[0])
likes=[]
for i in range(df.shape[0]):
    array_split=re.split(',', df['dish_liked'][i])
    for item in array_split:
        likes.append(item)


# In[58]:


df.index=range(df.shape[0])


# In[59]:


df.index


# In[60]:


print("count of most liked Dishes in Bangalore")
favourite_food=pd.Series(likes).value_counts()
favourite_food.head(40)


# In[61]:


ax=favourite_food.nlargest(n=40,keep='first').plot(kind='bar',figsize=(10,4), title="top 40 favourite food counts")


# ### Resturants and their counts

# In[62]:


plt.figure(figsize=(8,5))
restro=df['rest_type'].value_counts()[:20]
sns.barplot(restro,restro.index)
plt.title("Restaurant types")
plt.xlabel("count")


# ### Convert the online catogerical variable into numerical format 

# In[63]:


df.online_order[df.online_order=='Yes']=1
df.online_order[df.online_order=='No']=0


# In[64]:


df.online_order.value_counts()


# In[65]:


df.online_order=pd.to_numeric(df.online_order)


# ### Changing the string categories into int categories

# In[66]:


df.book_table[df.book_table=='Yes']=1
df.book_table[df.book_table=='No']=0


# In[67]:


df.book_table.value_counts()


# In[68]:


df.describe(include = "all")


# ### Insite
# 
# 1.There are 4379 unique restaurant names, of which Onesta has the highest occurrence  (85)
# 2.The most common restaurant type is "Casual Dining" (7326 occurrences)
# 3.The most common listed type is Delivery (10657)
#   Biryani is the most popular dish, but we can't be sure about this as dish_liked is missing over   half its data. 
# 4.There are 1681 unique levels in the cuisines column, this is because restaurants are             categorised under many different combinations of cuisines
# 5.Most popular cuisine  is North indian. It shows how Banglorians love north indian foods.  
# 6.Average cost for two at Bangalore restaurants is Rs 753 and there is very high variance           (standard deviation Rs 520)
# 7.Average number of votes per restaurant is 605. and here too there is high variance.
#   (standard deviation Rs 1113)
# 8.Majority of restaurants allow online ordering but don't allow online table booking
# 9.menu_item, reviews_list also contains many empty lists.
# 

# ### Label encode the categorical variable to make it easier to build Model

# In[69]:


#from sklearn.preprocessing import LableEncoder
#le= LabelEncoder()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[70]:


df.location=le.fit_transform(df.location)
df.rest_type=le.fit_transform(df.rest_type)
df.cuisines=le.fit_transform(df.cuisines)
df.menu_item=le.fit_transform(df.menu_item)


# In[71]:


df.head()


# In[72]:


my_data=df.iloc[:,[2,3,4,5,6,7,9,10,12]]
my_data.to_csv("zomato_df.csv")


# In[73]:


x=df.iloc[:,[2,3,5,6,7,9,10,12]]
x.head()


# In[74]:


y=df['rate'].values.reshape(-1,1)
y.ravel()


# In[75]:


df.hist(bins=40,figsize=(10,10))


# ### .heatmap()
# 
# For better insight i will plot heatmap.
# 
# The Big colorful picture below which is called Heatmap helps us to understand how features are correlated to each other. Postive sign implies postive correlation between two features whereas Negative sign implies negative correlation between two features.

# In[76]:


corr = df.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 10))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".3f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()


# ### One Visualization to Rule Them All
# We will perform analysis on the training data. The relationship between the features found in the training data is observed. In this way, inference about the properties can be made.
# 
# ### sns.pairplot
# Seaborn Pairplot uses to get the relation between each and every variable present in Pandas DataFrame. It works like a seaborn scatter plot but it plot only two variables plot and sns paiplot plot the pairwise plot of multiple features/variable in a grid format.

# In[77]:


sns.pairplot(df)
plt.show()


# In[78]:


# splitting the data set into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3, random_state=42)


# ### Stratified Sampling
# 
# This is a sampling technique that is best used when a statistical population can easily be broken down into distinctive sub-groups.Then samples are taken from each sub-groups based on the ratio of the sub groups size to the total population.

# In[79]:


print(f"Shape of X_train: {x_train.shape}\nShape of X_test: {x_test.shape}\nShape of y_train: {y_train.shape}\nShape of y_test: {y_test.shape}\n")


# In[80]:


x.head(5)


# In[81]:


#we will distribute the sample in test and train data so that no of zeros and ones could be same
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
split.get_n_splits(x, y)
for train_index,test_index in split.split(df,df[ 'online_order']):
    print("TRAIN:", train_index, "\nTEST:", test_index)
   
    strat_X_train = df.loc[train_index]
    strat_X_test = df.loc[test_index]


# ### Linear Regression

# In[82]:


from sklearn.metrics import r2_score
lr_model= LinearRegression()
lr_model.fit(x_train,y_train)
y_pred= lr_model.predict(x_test)
r2_score (y_test,y_pred)


# ### DecisionTreeRegressor

# In[83]:


# Fitting the Decision Tree Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regression_dt = DecisionTreeRegressor(random_state = 0)
regression_dt.fit(x_train, y_train)
y_pred=regression_dt.predict(x_test)
r2_score(y_test, y_pred)


# ### Random Forest

# In[84]:


from sklearn.ensemble import RandomForestRegressor
rf_model=RandomForestRegressor(n_estimators=700,random_state=42)
rf_model.fit(x_train,y_train)
y_predict=rf_model.predict(x_test)
r2_score(y_test,y_predict.ravel())


# ### ExtraTreeRegressor

# In[85]:


from sklearn.tree import ExtraTreeRegressor
extratree_model=ExtraTreeRegressor(random_state=42)
extratree_model.fit(x_train,y_train)
y_predict=extratree_model.predict(x_test)
r2_score(y_test,y_predict.ravel())


# ### Result
# 
# So from here we can conclude that out of multiple models RandomForestRegressor model is working well with 90.66% accuracy. which is a very good accuracy.

# In[86]:


# Using pickle we will save our model so that we can use it further
import pickle
pickle.dump(extratree_model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

