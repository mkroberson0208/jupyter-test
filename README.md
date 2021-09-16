# **Balance-Weighted Classification Models for Financial Products**

## 1.	Event indicators with balances

Financial cash flow models are frequently estimated on event indicators with a dollar balance attached:

*	Default indicator with balance charged-off the balance sheet
*	Prepayment indicator with balance prepaid
* Closure indicator with balance removed due to attrition 

Although the event occurrence could be estimated as a 0/1 binary classification task, this could misrepresent the target variable of interest. For example, if larger loan sizes are more likely to default, the dollar charge-off rate will be above the default event incidence rate. Likewise, often times smaller deposit account balances are more likely to close, and the dollar closure rate is below the event rate. In either case the model estimated on event indicator will be misspecified and produce incorrect forecasts. The balance-weighted rates reflect the actual cash flow and impact on financial performance.


## 2. Adjustment to balance
There are two possible steps to adapt a classification model (i.e. logistic regression) for event indicators with dollar balances:

1. Include balance as an input variable in the model
2. Re-weight the event indicator to match the dollar balance impact

The second method is preferable because it doesn’t necessarily assume a fixed relationship between balance and event occurrence. Directly re-weighting observations instead uses the observed position-level relationship in each time period, which might change with the economic cycle. 

Weighting observations by dollar balance is also common in investment portfolio management, where each position receives a weight of % dollar allocation. A portfolio of loans or savings products is no different in this regard. The weight for each loan-level or account-level observation would be % of balance for that product class at observation date (t):
w_i=B_i/(∑_(i=1)^N▒B_i )

The weights can be passed to a classification model as sample weight parameters. This effectively over-samples or under-samples each observation depending on whether they have a higher-than-average or lower-than-average balance, respectively.
