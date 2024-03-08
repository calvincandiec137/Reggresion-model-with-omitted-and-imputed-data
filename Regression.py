# =============================================================================
# PREDICTING PRICE OF PRE-OWNED CARS 
# =============================================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt


# =============================================================================
# Setting dimensions for plot 
# =============================================================================

sns.set(rc={'figure.figsize':(11.7,8.27)})

# =============================================================================
# Reading CSV file
# =============================================================================

cars_data=pd.read_csv('cars_sampled.csv')

cars=cars_data.copy()

cars.info()

# =============================================================================
# Summarizing data
# =============================================================================

cars.describe()

pd.set_option('Display.float_format',lambda x:'%3.f' %x)
cars.describe()

pd.set_option('display.max_columns',500)
cars.describe

# =============================================================================
# Dropping unwanted columns
# =============================================================================
cols=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=cols,axis=1)

# =============================================================================
# Removing duplicate records
# =============================================================================

cars.drop_duplicates(keep='first',inplace=True)

# =============================================================================
# Data cleaning
# =============================================================================
cars.isnull().sum()

yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()

sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)

price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'],bins=10)
cars['price'].describe()
sns.boxplot(cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)

power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'],bins=10)
cars['powerPS'].describe()


# =============================================================================
# Working range of data
# =============================================================================

cars = cars[
        (cars.yearOfRegistration <= 2018) 
      & (cars.yearOfRegistration >= 1950) 
      & (cars.price >= 100) 
      & (cars.price <= 150000) 
      & (cars.powerPS >= 10) 
      & (cars.powerPS <= 500)]

cars['monthOfRegistration']=cars['monthOfRegistration']/12

cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['temp']=cars['Age'].copy()
cars['Age']=cars['yearOfRegistration']
cars['yearOfRegistration']=cars['temp']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()

cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration','temp'],)


# Visualizing parameters 

# Age
sns.displot(cars['Age'],bins=10)
sns.boxplot(y=cars['Age'])

# price
sns.displot(cars['price'],bins=10)
sns.boxplot(y=cars['price'])

# powerPS
sns.displot(cars['powerPS'],bins=10)
sns.boxplot(y=cars['price'])

#Price vs Age

sns.regplot(x='Age',y='price',data=cars,scatter=True,fit_reg=False)

# powerPS vs price
sns.regplot(x='powerPS', y='price', scatter=True,fit_reg=False, data=cars)

#Variable Seller 
cars['seller'].value_counts()
pd.crosstab(cars['seller'], columns='count')
#Since there is only one commercial type and rest private hence seller doesnt affect price much

#variable offerType
cars['offerType'].value_counts()
pd.crosstab(cars['offerType'], columns='count')
    
# Variable abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x= 'abtest',data=cars)

# Variable vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x= 'vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)

# Variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x= 'gearbox',data=cars)
sns.boxplot(x= 'gearbox',y='price',data=cars)
# gearbox affects price 


# Variable model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x= 'model',data=cars)
sns.boxplot(x= 'model',y='price',data=cars)
# Cars are distributed over many models
# Considered in modelling


# Variable kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.boxplot(x= 'kilometer',y='price',data=cars)
cars['kilometer'].describe()
sns.distplot(cars['kilometer'],bins=8 ,kde=False)
sns.regplot(x='kilometer', y='price', scatter=True, 
            fit_reg=False, data=cars)
# Considered in modelling


# Variable fuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x= 'fuelType',data=cars)
sns.boxplot(x= 'fuelType',y='price',data=cars)
# fuelType affects price

# Variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x= 'brand',data=cars)
sns.boxplot(x= 'brand',y='price',data=cars)
# Cars are distributed over many brands
# Considered for modelling 

# Variable notRepairedDamage
# yes- car is damaged but not rectified
# no- car was damaged but has been rectified
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x= 'notRepairedDamage',data=cars)
sns.boxplot(x= 'notRepairedDamage',y='price',data=cars)
# As expected, the cars that require the damages to be repaired
# fall under lower price ranges



# =============================================================================
# Removing insignificant variables
# =============================================================================

col=['seller','offerType','abtest']
cars=cars.drop(columns=col, axis=1)#cleaned data
cars_copy=cars.copy()

# =============================================================================
# Correlation
# =============================================================================

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values()[1:]

# =============================================================================
# OMITTING MISSING VALUES
# =============================================================================

cars_omitted=cars.dropna(axis=0)
cars_omitted=pd.get_dummies(cars_omitted,drop_first=True)

# =============================================================================
# IMPORTING NECESSARY LIBRARIES
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# =============================================================================
# MODEL BUILDING WITH OMITTED DATA
# =============================================================================

# Separating input and output features
x1 = cars_omitted.drop(['price'], axis='columns', inplace=False)
y1 = cars_omitted['price']

# Plotting the variable price
prices = pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
prices.hist()

# Transforming price as a logarithmic value
y1 = np.log(y1)

# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state = 3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# =============================================================================
# BASELINE MODEL FOR OMITTED DATA
# =============================================================================

"""
We are making a base model by using test data mean value
This is to set a benchmark and to compare with our regression model
"""

# finding the mean for test data value
base_pred = np.mean(y_test)
print(base_pred)

# Repeating same value till length of test data
base_pred = np.repeat(base_pred, len(y_test))

# finding the RMSE
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
                               
print(base_root_mean_square_error)

# =============================================================================
# LINEAR REGRESSION WITH OMITTED DATA
# ======================================================================


lgr=LinearRegression(fit_intercept=True)

model_lin1=lgr.fit(X_train,y_train)

cars_prediction_lin1=lgr.predict(X_test)

lin_mse1=mean_squared_error(y_test,cars_prediction_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

residual1=y_test-cars_prediction_lin1
sns.regplot(x=cars_prediction_lin1,y=residual1,scatter=True,fit_reg=False)
residual1.describe()

# =============================================================================
# RANDOM FOREST WITH OMITTED DATA
# =============================================================================

# Model parameters
rf = RandomForestRegressor(n_estimators = 100,
                           max_depth=100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)

# Model
model_rf1=rf.fit(X_train,y_train)

# Predicting model on test set
cars_predictions_rf1 = rf.predict(X_test)

# Computing MSE and RMSE
rf_mse1 = mean_squared_error(y_test, cars_predictions_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)

# R squared value
r2_rf_test1=model_rf1.score(X_test,y_test)
r2_rf_train1=model_rf1.score(X_train,y_train)
print(r2_rf_test1,r2_rf_train1)

# =============================================================================
# MODEL BUILDING WITH IMPUTED DATA
# =============================================================================   

cars_imputed=cars.apply(lambda x:x.fillna(x.median())\
                        if x.dtype=='float'else \
                        x.fillna(x.value_counts().index[0]))

cars_imputed.isnull().sum()

cars_imputed=pd.get_dummies(data=cars_imputed,drop_first=True)


# =============================================================================
# MODEL BUILDING WITH IMPUTED DATA
# =============================================================================

y2=cars_imputed['price']
x2=cars_imputed.drop(['price'],axis='columns',inplace=False)

prices=pd.DataFrame({"1. Before":y2,"2. After":np.log(y2)})

y2=np.log(y2)

x_train1,x_test1,y_train1,y_test1=train_test_split(x2,y2,random_state=3,test_size=0.3)
print(x_train1.shape, x_test1.shape, y_train1.shape, y_test1.shape)
# =============================================================================
# BASELINE MODEL FOR IMPUTED DATA
# =============================================================================
# finding the mean for test data value
base_pred = np.mean(y_test1)
print(base_pred)

# Repeating same value till length of test data
base_pred = np.repeat(base_pred, len(y_test1))

# finding the RMSE
base_root_mean_square_error_imputed = np.sqrt(mean_squared_error(y_test1, base_pred))
                               
print(base_root_mean_square_error_imputed)


# =============================================================================
# LINEAR REGRESSION WITH IMPUTED DATA
# =============================================================================
# Setting intercept as true
lgr2=LinearRegression(fit_intercept=True)

# Model
model_lin2=lgr2.fit(x_train1,y_train1)

# Predicting model on test set
cars_predictions_lin2 = lgr2.predict(x_test1)

# Computing MSE and RMSE
lin_mse2 = mean_squared_error(y_test1, cars_predictions_lin2)
lin_rmse2 = np.sqrt(lin_mse2)
print(lin_rmse2)

# R squared value
r2_lin_test2=model_lin2.score(x_test1,y_test1)
r2_lin_train2=model_lin2.score(x_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)

# =============================================================================
# RANDOM FOREST WITH IMPUTED DATA
# =============================================================================

# Model parameters
rf2 = RandomForestRegressor(n_estimators = 100,
                           max_depth=100,min_samples_split=10,
                           min_samples_leaf=4,random_state=1)

# Model
model_rf2=rf2.fit(x_train1,y_train1)

# Predicting model on test set
cars_predictions_rf2 = rf2.predict(x_test1)

# Computing MSE and RMSE
rf_mse2 = mean_squared_error(y_test1, cars_predictions_rf2)
rf_rmse2 = np.sqrt(rf_mse2)
print(rf_rmse2)

