# **Balance-Weighted Class Prediction for Financial Products**

## 1.	Event indicators with balances

Financial cash flow models are frequently estimated on event indicators with a dollar-denominated balance attached:

* Default indicator with balance charged-off the balance sheet
* Prepayment indicator with balance prepaid
* Closure indicator with balance removed due to attrition 

Although a binary classification model could be estimated from event indicator alone, this would misrepresent the target variable of interest, the impact to portfolio balance. For example, if larger loan sizes are more likely to default, the dollar gross loss or charge-off rate will be consistenly above the default event rate. Likewise, often times smaller account balances are more likely to close, and the dollar-denominated closure rate is below the event rate. In either case the classification model estimated on event indicator will be misspecified and produce incorrect forecasts. Balance-weighted rates reflect the actual cash flow and impact on enterprise-level financial performance.


## 2. Adjustment to portfolio weight
There are two possible steps to adapt a classification model (i.e. logistic regression) for event indicators with dollar balances:

1. Include balance as an input variable in the model
2. Re-weight the event indicator to dollar balance

The second method is preferable because it doesn’t necessarily assume a fixed relationship between balance and event occurrence. Directly re-weighting observations instead uses the observed position-level relationship in each time period. 

Weighting observations by dollar balance is also common in investment portfolio management, where each position receives a weight of % dollar allocation. A portfolio of loans or savings products is no different in this regard. The weight for each loan-level or account-level observation would be % of observation-level balance to total balance:

w<sub>i</sub>=B<sub>i</sub>/(∑B<sub>i</sub>)

The weights are then passed to a classification model as sample weight parameters. This effectively over-samples or under-samples each observation depending on whether they have a higher-than-average or lower-than-average balance, respectively.

## 3. Example

```python
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

df = pd.read_csv('https://raw.githubusercontent.com/mkroberson0208/jupyter-test/main/synthetic_data.csv')
df = df.sort_values('Default_ind')
df.reset_index(inplace=True,drop=True)
df.index = df.index + 1
print(df)
```

        Face_amt  Default_ind   x1  x2
    1         50            0  0.8   8
    2         50            0  1.0   5
    3         50            0  6.0   6
    4         50            0  3.5   1
    5         50            0  2.0   2
    6         50            0  0.5   1
    7         50            0  4.0   2
    8         50            0  3.0   3
    9         50            0  2.5   4
    10       200            1  5.0   6
    11        50            1  8.0   2
    


```python
# Default rate = event incidence (0/1 indicator)
# Loss or charge-off rate = balance lost (0/1 multiplied by $ balance)
d = {}
d['Default Rate'] = df['Default_ind'].sum()/df['Default_ind'].count()
d['Charge-off Rate'] = (df['Face_amt']*df['Default_ind']).sum()/df['Face_amt'].sum()
print(d)
```

    {'Default Rate': 0.18181818181818182, 'Charge-off Rate': 0.35714285714285715}
    


```python
# weight = % of portfolio balance per row
# Similar to portfolio weights in caluclating risk/return for investments
df['weight'] = df['Face_amt']/df['Face_amt'].sum()

# Logistic model 
#   - Estimated on default event indicator to output default rate

iv = ['x1']
x = np.asarray(df[iv])
y = np.asarray(df['Default_ind'])
model = sm.GLM(endog=y, exog=sm.add_constant(x), family=sm.families.Binomial())
result = model.fit()
print(result.summary())
df['y_hat'] = result.predict(sm.add_constant(x))
print('Predicted charge-off rate: ', (df['Face_amt']*df['y_hat']).sum()/df['Face_amt'].sum())
```

                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                   11
    Model:                            GLM   Df Residuals:                        9
    Model Family:                Binomial   Df Model:                            1
    Link Function:                  logit   Scale:                          1.0000
    Method:                          IRLS   Log-Likelihood:                -2.4147
    Date:                Thu, 16 Sep 2021   Deviance:                       4.8294
    Time:                        08:24:40   Pearson chi2:                     4.21
    No. Iterations:                     7                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -6.9947      4.397     -1.591      0.112     -15.614       1.624
    x1             1.2060      0.827      1.458      0.145      -0.416       2.828
    ==============================================================================
    Predicted charge-off rate:  0.20198735515433805
    


```python
# Balance-weighted logistic model
#   - Estimated on default event indicator re-weighted by % balance to output loss rate
iv = ['x1']
x = np.asarray(df[iv])
y = np.asarray(df['Default_ind'])
model = sm.GLM(endog=y, exog=sm.add_constant(x), family=sm.families.Binomial(),freq_weights=np.asarray(df['weight']))
result = model.fit()
print(result.summary())
df['y_hat'] = result.predict(sm.add_constant(x))
print('Predicted charge-off rate: ', (df['Face_amt']*df['y_hat']).sum()/df['Face_amt'].sum())
```

                     Generalized Linear Model Regression Results                  
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                   11
    Model:                            GLM   Df Residuals:                       -1
    Model Family:                Binomial   Df Model:                            1
    Link Function:                  logit   Scale:                          1.0000
    Method:                          IRLS   Log-Likelihood:               -0.33182
    Date:                Thu, 16 Sep 2021   Deviance:                      0.66364
    Time:                        09:10:31   Pearson chi2:                    0.772
    No. Iterations:                     6                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -6.9411     15.563     -0.446      0.656     -37.445      23.562
    x1             1.4986      3.324      0.451      0.652      -5.016       8.013
    ==============================================================================
    Predicted charge-off rate:  0.3571428572928815
    
## Conclusion

In financial cash flow analysis the model output is typically class likelihood multiplied by principal balance. This adjusts the balance in each time period by amount lost to credit defaults, prepayments, or closures. In order for predicted class likelihood to reflect the underlying financial impact, model estimation on event indicator must be re-scaled by observation-level principal balance divided by portfolio balance, or portfolio weight. Balance-weighting helps to produce accurate forecasts in dollar terms while retaining the use of well-established classification models such as logistic regression.
