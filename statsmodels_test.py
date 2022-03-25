import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression

import pandas as pd

df = pd.read_csv('sourcefile.csv')
y = df['species']
x = df.drop(['species'], axis=1)

y = y.astype('category').cat.codes

x_with_const = sm.add_constant(x)
statsmodels_linear = sm.OLS(y, x_with_const).fit()
sklearn_linear = LinearRegression().fit(x, y)

print(statsmodels_linear.params)
print(sklearn_linear.coef_)

statsmodels_regularized = sm.OLS(y, x_with_const).fit_regularized(alpha=1, L1_wt=0.8, method='elastic_net')
sklearn_regularized = ElasticNet(alpha=1, l1_ratio=0.8).fit(x, y)

print(statsmodels_regularized.params)
print(sklearn_regularized.coef_)



