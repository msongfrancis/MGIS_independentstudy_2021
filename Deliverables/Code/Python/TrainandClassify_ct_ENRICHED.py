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

year = "2018"
file_path = f"C:/Users/msong/Desktop/Independent proj/tripcounts_{year}_enriched.csv"

cols = ['GISJOINID', 
        'year',
        'SUM_TripCount',
        'populationtotals_totpop_cy',
        'populationtotals_popdens_cy', 
        'householdincome_medhinc_cy_i', 
        'raceandhispanicorigin_divindx_cy',
        'educationalattainment_hsgrad_cy_p',
        'educationalattainment_asscdeg_cy_p',
        'educationalattainment_bachdeg_cy_p',
        'educationalattainment_graddeg_cy_p', 
        'households_acshhbpov_p',
        'employmentunemployment_unemprt_cy', 
        'raceandhispanicorigin_white_cy_p']
df = pd.read_csv(file_path,usecols=cols)

df['householdincome_medhinc_cy_i'].describe()

# create df that do not correspond with a census tract
empty_df = df.loc[(df['GISJOINID'].isna())]

# dataset with no null vals
data = df[df['GISJOINID'].notna()].drop('GISJOINID',axis=1).astype(int)


data['SUM_TripCount'].describe()

corr = data.astype('float64').corr()

# split data into test and train set for validation
# fracnum is the percentage
fracNum = 0.30
train_set = data.sample(frac = fracNum)
test_set = data.drop(train_set.index)

# demographic fields to serve as input values in input
x_cols = ['populationtotals_totpop_cy',
          'populationtotals_popdens_cy', 
          'householdincome_medhinc_cy_i',
          'raceandhispanicorigin_divindx_cy', 
          'educationalattainment_hsgrad_cy_p',
          'educationalattainment_asscdeg_cy_p',
          'educationalattainment_bachdeg_cy_p',
          'educationalattainment_graddeg_cy_p', 
          'households_acshhbpov_p',
          'employmentunemployment_unemprt_cy',
          'raceandhispanicorigin_white_cy_p']

# field to predict
y_cols = 'SUM_TripCount'

# create training sets
X_train = train_set[x_cols]
y_train = train_set[y_cols]           
              

# create test set
X_test = test_set[x_cols]
y_test = test_set[y_cols]

# run linear regression
lin_reg = LinearRegression()
_ = lin_reg.fit(X_train, y_train) # train and compute regression
preds = lin_reg.predict(X_test) # predict scores

print("Training score:", lin_reg.score(X_train, y_train))
print("Testing score:", lin_reg.score(X_test, y_test))
print("MAE of Linear Regression:", mean_absolute_error(y_test, preds), '\n')

# based on data, ridge is not a good method because data is not have multicollinearity
ridge = Ridge(alpha=0.1) # alpha can be altered
_ = ridge.fit(X_train, y_train) # train and compute regression
preds = ridge.predict(X_test) # predict scores

print("Training score:", ridge.score(X_train, y_train))
print("Testing score:", ridge.score(X_test, y_test))
print("MAE of Ridge Regression:", mean_absolute_error(y_test, preds), '\n')


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
clf=RandomForestClassifier(n_estimators=100,bootstrap=False,warm_start=True)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(list(y_test), y_pred))

# Train and run random forest regression
rf_regr = RandomForestRegressor(n_estimators = 1000, random_state=0)
_ = rf_regr.fit(X_train, y_train) # create trees in forest
preds=rf_regr.predict(X_test) # predict values

print("Training score:", rf_regr.score(X_train, y_train))
print("Testing score:", rf_regr.score(X_test, y_test))
print("MAE of Random Forest Regression:", mean_absolute_error(y_test, preds), '\n')


