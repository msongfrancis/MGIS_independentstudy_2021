"""
Train and Classify - Census Tracts

This script explores different machine learning regressions to predict
total e-scooter trip counts for census tracts in Minneapolis MN. The
demographic data are from ACS-surveys for 2014-2018, and 2015-2019.
The regressions explored are Random Forest, Linear, Ridge and Poisson regression

The script requires sklearn, pandas, and an ArcGIS pro license.

Data sources: ACS-Survey 2014-2018 5-year Estimates, ACS-Survey 2015-2019
5-year Estimates, City of Minneapolis, U.S. Census Bureau
"""

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

file_path = r"C:\Users\msong\Desktop\Independent proj\escooter_ML\escooter_all.csv"
cols = ['GISJOIN', 
        'year', 
        'SUM_TripCount', 
        'percent_nonwhite',
       'percent_hsandabv', 
        'medhhinc_normal',
       'popdens_sqmi']
#        'med_hh_inc']
df = pd.read_csv(file_path,usecols=cols)

# create df that do not correspond with a census tract
empty_df = df.loc[(df['GISJOIN'].isna())]

# dataset with no null vals
data = df[df['GISJOIN'].notna()].drop('GISJOIN',axis=1)

data['SUM_TripCount'].describe()

# pearson's correlation to see relationships between variables
# ignore year correlation values
corr = data.astype('float64').corr()

corr

# split data into test and train set for validation
# fracnum is the percentage
fracNum = 0.30
train_set = data.sample(frac = fracNum)
test_set = data.drop(train_set.index)

# input demographic fields of interest
x_cols = ['percent_nonwhite',
          'percent_hsandabv', 
          'medhhinc_normal',
          'popdens_sqmi']
# field to predict
y_cols = 'SUM_TripCount'


# indicate fields to be used in multilinear regression
X_train = train_set[x_cols]
y_train = train_set[y_cols]         

# format test set
X_test = test_set[x_cols]
y_test = test_set[y_cols]



# run linear regression
lin_reg = LinearRegression()
_ = lin_reg.fit(X_train, y_train) # train regression with training set
preds = lin_reg.predict(X_test) # predict values in test set

print("Training score:", lin_reg.score(X_train, y_train))
print("Testing score:", lin_reg.score(X_test, y_test))
print("MAE of Linear Regression:", mean_absolute_error(y_test, preds), '\n')

# based on pearson's correlation, ridge is not a good method 
# because data is not have multicollinearity
ridge = Ridge(alpha=0.1) # alpha can be altered
_ = ridge.fit(X_train, y_train)
preds = ridge.predict(X_test)

print("Training score:", ridge.score(X_train, y_train))
print("Testing score:", ridge.score(X_test, y_test))
print("MAE of Ridge Regression:", mean_absolute_error(y_test, preds), '\n')

# Resource: # https://www.datacamp.com/community/tutorials/random-forests-classifier-python
clf=RandomForestClassifier(n_estimators=100,
                           bootstrap=True,
                           warm_start=True,
                           max_features="sqrt"
                           # oob_score=True
                           #min_samples_split=.1
                           # random_state = 0
                           #
                          ) 

# Accuracy score will not compute anything when I have bootstrap set to True
# Have tried reducing fields for random forest

# Other factors tried:
# warm_start=True
# min_samples_split=10
# random_state = 0

_= clf.fit(X_train,y_train) # create branches and trees from training data
y_pred=clf.predict(X_test) # predict values in test dataset

print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # check accuracy score of test and train.

# run Random Forest Regressor
rf_regr = RandomForestRegressor(n_estimators = 1000, random_state=0)
_ = rf_regr.fit(X_train, y_train) # create trees in forest
preds=rf_regr.predict(X_test) # predict values in test set

print("Training score:", rf_regr.score(X_train, y_train))
print("Testing score:", rf_regr.score(X_test, y_test))
print("MAE of Random Forest Regression:", mean_absolute_error(y_test, preds), '\n')

""" 
resources:
- about: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html
- about_2: https://timeseriesreasoning.com/contents/poisson-regression-model/
- tutorial: https://www.kaggle.com/gauravduttakiit/explore-the-poisson-regression
"""


# pr = PoissonRegressor()
# pr.fit(X_train, y_train)
# y_pr = pr.predict(X_test)"

# reformat table to include two fields:
# year of dataset and total trip counts
pdata = data[["year","SUM_TripCount"]]

# create a training and test set
train,test=train_test_split(pdata, train_size = .3,random_state =1)

# reshape SUM_TripCount to be from a scale of -1 to 1
X_train = train['SUM_TripCount'].values.reshape(-1, 1)
y_train = train.year
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)


X_train.shape,y_train.shape

# reshape SUM_TripCount to be from a scale of -1 to 1
X_test = test['SUM_TripCount'].values.reshape(-1, 1)
y_test = test.year
X_test.shape,y_test.shape

# Train regression and predict trip counts
pipeline = Pipeline([('standardscaler', StandardScaler()),('model', PoissonRegressor())])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

r2_test = metrics.r2_score(y_test, y_pred)
r2_test



"""
Notes: 

Documentation:
<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random%20forest#sklearn.ensemble.RandomForestClassifier.score>

- warm_start: bool, default=False
When set to True, reuse the solution of the previous call to fit and 
add more estimators to the ensemble, otherwise, just fit a whole new forest. 
See the Glossary.

- bootstrap : bool, default=True
Whether bootstrap samples are used when building trees. 
If False, the whole dataset is used to build each tree.

- max_features{“auto”, “sqrt”, “log2”}, int or float, default=”auto”
The number of features to consider when looking for the best split:

- random_state : int, RandomState instance or None, default=None
Controls both the randomness of the bootstrapping of the samples used 
when building trees (if bootstrap=True) and the sampling of the 
features to consider when looking for the best split at each node 
(if max_features < n_features). See Glossary for details.

- min_samples_leaf : int or float, default=1
The minimum number of samples required to be at a leaf node. 
A split point at any depth will only be considered if it leaves at 
least min_samples_leaf training samples in each of the left and right 
branches. This may have the effect of smoothing the model, especially 
in regression.

"""


