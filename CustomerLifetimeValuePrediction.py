import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.compat import lzip
import xgboost as xg

# Read in the file
Clv = pd.read_csv("FutureMargin1.csv")


# Exploration of the sales data
print(Clv.head(10))
type(Clv)

# Data Understanding
    # id: identification number of customer
    # nOrders: Number of orders by the customer
    # nItems: Total number of items ordered by the customers across order; each order may have multiple items
    # daysSinceLastOrder: Number of days since the last order
    # margin: Mean margin aggregated across orders
    # returnRatio: ratio of orders returned by the customers to total orders
    # shareOwnBrand: Percentage of private label
    # shareVoucher: Sales from voucher
    # shareSale: Voucher related sales
    # gender, age: Demographics
    # marginPerOrder: Margin per order aggregated
    # marginPerItem: margin per item aggregated
    # itemsPerOrder: mean items per order
    # futureMargin: Target variable


# Checking the data type across the data set

Clv.dtypes
    # The data types across variables. Only categorical variable is Gender.


# Check if any values are missing
Clv.isnull().values.any()
        # No missing data in any of the variables


# correlation of the target variable - future margin with other variables in the dataset
Clv.corr()["futureMargin"]
    # Margin has the highest correlation with future margin
    # Number of orders and number of items are highly correlated with future margin
    # Interesting to note that margin per order and margin per item is negatively correlated with future margin


# histogram of margin per order

plt.figure()

%matplotlib inline
plt.hist(Clv['marginPerOrder'], bins=20)

# Negative correlation because the number of order and items (positively correlated with future margins) are in the denominator



# Multi-collinearity

Clv.corr()

    # Number of orders and number of items are highly correlated
    # Correlation between margin and number of orders and items are also fairly high

# calculate VIF
Clv = Clv.drop(['gender'], axis=1)
Clv = Clv.drop(['customerID'], axis=1)

# Split the data into training/testing sets
clv_X_train = Clv[:-20]
clv_X_test = Clv[-20:]


# Split the targets into training/testing sets
clv_y_train = Clv.futureMargin[:-20]
clv_y_test = Clv.futureMargin[-20:]


clv_X_train = clv_X_train.drop('futureMargin', axis=1)
clv_X_test = clv_X_test.drop('futureMargin', axis=1)


# Create linear regression object
regr = sk.linear_model.LinearRegression()

clv_X_train.head(5)




# Train the model using the training sets
regr.fit(clv_X_train, clv_y_train)

# Make predictions using the testing set
clv_y_pred = regr.predict(clv_X_test)



for idx, col_name in enumerate(clv_X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regr.coef_[idx]))


# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(clv_y_test, clv_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(clv_y_test, clv_y_pred))


# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(Clv.values, i) for i in range(Clv.shape[1])]
vif["features"] = Clv.columns
vif.round()


# Dropping nItems as it is heavily correlated with nOrders

Clv1 = Clv.drop('futureMargin', axis=1)
Clv2 = Clv1.drop('nItems', axis=1)

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(Clv2.values, i) for i in range(Clv2.shape[1])]
vif["features"] = Clv2.columns
vif.round()


# Removing marginperorder, marginperitem, and itemsperorder as they are all highly correlated with margins or orders; decided to retain raw margins and take off the margin related calculated fields

Clv3 = Clv2.drop('marginPerOrder', axis=1)
Clv4 = Clv3.drop('marginPerItem', axis=1)
Clv5 = Clv4.drop('age', axis=1)
Clv6 = Clv5.drop('nOrders', axis=1)

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(Clv6.values, i) for i in range(Clv6.shape[1])]
vif["features"] = Clv6.columns
vif.round()


# Checking assumptions of normality
Clv.head(5)
model = smf.ols("futureMargin ~ daysSinceLastOrder + margin + returnRatio +  shareOwnBrand + shareVoucher + shareSale + itemsPerOrder", data = Clv).fit()
model.summary()


stats.probplot(model.resid, plot= plt)
plt.title("Model1 Residuals Probability Plot")

# Residuals are normally distributed! Woot! Hence inference tests can be used


# Homoscedasticity or constant variance of residuals

TestNames = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(model.resid, model.model.exog)
lzip(TestNames, test)


# Split the data into training/testing sets
clv_X_train = Clv6[:-20]
clv_X_test = Clv6[-20:]


# Split the targets into training/testing sets
clv_y_train = Clv.futureMargin[:-20]
clv_y_test = Clv.futureMargin[-20:]


# Create linear regression object
regr = sk.linear_model.LinearRegression()

clv_X_train.head(5)



# Train the model using the training sets
regr.fit(clv_X_train, clv_y_train)

# Make predictions using the testing set
clv_y_pred = regr.predict(clv_X_test)

for idx, col_name in enumerate(clv_X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regr.coef_[idx]))


# R squared
print("R squared: %.2f"
       % regr.score(clv_X_test, clv_y_test))

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(clv_y_test, clv_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(clv_y_test, clv_y_pred))


# making predictions
ClvPredict = pd.read_csv('FutureMargin2.csv')
ClvPredict = ClvPredict.drop(['customerID','nOrders','nItems','gender','marginPerOrder','marginPerItem','age'], axis=1)
ClvPredict.head(10)
type(ClvPredict)
ClvPredict.dtypes


# Missing values
ClvPredict.isnull().values.any()
ClvPredict = ClvPredict.dropna()


ClvPredict.isnull().values.any()


# Testing before passing through predict
ClvPredict.shape


# Calculating the model output as per prediction

ClvResults = regr.predict(ClvPredict)
ClvResults = pd.DataFrame(ClvResults)
ClvResults.head(10)



##################################################################
### XG Boost for predicting the future margin#####################
##################################################################

# Read in the file
ClvXg = pd.read_csv("FutureMargin1.csv")
ClvXg.head(10)
ClvXg1 = ClvXg.drop(['gender'], axis=1)
ClvXg2 = ClvXg1.drop(['customerID'], axis=1)



# Creating train and test data set
X, y = ClvXg2.iloc[:,:-1],ClvXg2.iloc[:,-1]

X_train, X_test, y_train, y_test= train_test_split(X, y,
        test_size=0.2, random_state=123)


X_train.head(2)


# Running the XG Boost model

import xgboost as xgb

xg_reg = xgb.XGBRegressor(objective='reg:linear',n_estimators=25, seed=123)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

# Results

rmse = np.sqrt(mean_squared_error(y_test,preds))

print("RMSE: %f" % (rmse))


# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()





