"""
## Train and Classify - Enriched Block Groups

This script explores different machine learning regressions to predict
total e-scooter trip counts for census block groups in Minneapolis MN.
The demographic data is sourced from ESRI Enrich tool. The regressions
explored are Random Forest, Linear, and Ridge regression. 

The script requires sklearn, pandas, and an ArcGIS pro license.

Data sources: ACS-Survey 2014-2018 5-year Estimates, ACS-Survey 2015-2019
5-year Estimates, City of Minneapolis, U.S. Census Bureau
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import os

file_path = f"C:/Users/msong/Desktop/Independent proj/mpls_blkgrps_2010_join.csv"

# fields for demographic data to be used in regressions
cols = ['geoid_text',
        'populationtotals_totpop_cy',
        'populationtotals_popdens_cy', 
        'householdincome_medhinc_cy_i',
        'foodstampssnap_acssnap_p', 
        'raceandhispanicorigin_divindx_cy',
        'educationalattainment_acssomehs_p', 
        'atrisk_acshhbpov_p',
        'trip_count_start', 
        'trip_count_end'
       ]

df = pd.read_csv(file_path,usecols=cols)

# statistics about fields
df["geoid_text"].describe()

# create df that do not correspond with a census tract
empty_df = df.loc[(df['trip_count_start'].isna())]

# dataset with no null vals
data = df[df['trip_count_start'].notnull()]

# stats about predicting field after dropping nulls
data['trip_count_start'].describe()

# split data into test and train set for validation
# fracnum is the percentage of whole
fracNum = 0.30
train_set = data.sample(frac = fracNum)
test_set = data.drop(train_set.index)

# demographic fields to use as indicators
x_cols = ['geoid_text',
        'populationtotals_totpop_cy',
        'populationtotals_popdens_cy', 
        'householdincome_medhinc_cy_i',
        'foodstampssnap_acssnap_p', 
        'raceandhispanicorigin_divindx_cy',
        'educationalattainment_acssomehs_p', 
        'atrisk_acshhbpov_p']

# field to predict from regressions
y_cols = 'trip_count_start'

# indicate fields to be used in multilinear regression
trainID = train_set['geoid_text']
X_train = train_set[x_cols].drop('geoid_text',axis=1).copy()
y_train = train_set[y_cols]    
              
# format test set
testID = test_set['geoid_text'] # unique identifier of the test set
X_test = test_set[x_cols].drop('geoid_text',axis=1).copy()
y_test = test_set[y_cols]

# remove geoid_text because it will not be used as a value to 
# help predict total escooter trips
x_cols.remove("geoid_text")

# run random forest regression
rf_regr = RandomForestRegressor(n_estimators = 1000, random_state=0)
_ = rf_regr.fit(X_train, y_train) # create and train trees
rf_preds=rf_regr.predict(X_test) # predict values in X_test dataset

print("Training score:", rf_regr.score(X_train, y_train))
print("Testing score:", rf_regr.score(X_test, y_test))
print("MAE of Random Forest Regression:", mean_absolute_error(y_test, rf_preds), '\n')



# check collinearity between variables in dataset
corr = data.astype('float64').corr()

# run linear regression
lin_reg = LinearRegression()
_ = lin_reg.fit(X_train, y_train)
lr_preds = lin_reg.predict(X_test)

print("Training score:", lin_reg.score(X_train, y_train)) # r-squared
print("Testing score:", lin_reg.score(X_test, y_test))
print("MAE of Linear Regression:", mean_absolute_error(y_test, lr_preds), '\n')

# run ridge linear regression
# based on data, ridge is not a good method because data is not have multicollinearity
ridge = Ridge(alpha=0.1) # alpha can be altered
_ = ridge.fit(X_train, y_train)
r_preds = ridge.predict(X_test)

print("Training score:", ridge.score(X_train, y_train))
print("Testing score:", ridge.score(X_test, y_test))
print("MAE of Ridge Regression:", mean_absolute_error(y_test, r_preds), '\n')


# create results table with the predicted vals from each regression
# add geoid to join with spatial data
results = X_test

results["geoid"] = testID
results["test_counts"] = y_test
results["rf_predicted"] = rf_preds
results["lr_predicted"] = lr_preds
results["r_predicted"]= r_preds

outpath=r"C:\Users\msong\Desktop\Independent proj"
results.to_csv(os.path.join(outpath,"results.csv"),index=False)


