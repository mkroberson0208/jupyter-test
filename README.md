# **Balance-Weighted Classification Models for Financial Products**

## 1.	Event indicators with balances

Financial cash flow models are frequently estimated on event indicators with a dollar balance attached:

*	Default indicator with balance charged-off the balance sheet
*	Prepayment indicator with balance prepaid
* Closure indicator with balance removed due to attrition 

Although the event occurrence could be estimated as a 0/1 binary classification task, this could misrepresent the target variable of interest. For example, if larger loan sizes are more likely to default, the dollar charge-off rate will be above the default event incidence rate. Likewise, often times smaller deposit account balances are more likely to close, and the dollar closure rate is below the event rate. In either case the model estimated on event indicator will be misspecified and produce incorrect forecasts. The balance-weighted rates reflect the actual cash flow and impact on financial performance.


## 2. Adjustment to portfolio weights
There are two possible steps to adapt a classification model (i.e. logistic regression) for event indicators with dollar balances:

1. Include balance as an input variable in the model
2. Re-weight the event indicator to match the dollar balance impact

The second method is preferable because it doesn’t necessarily assume a fixed relationship between balance and event occurrence. Directly re-weighting observations instead uses the observed position-level relationship in each time period, which might change with the economic cycle. 

Weighting observations by dollar balance is also common in investment portfolio management, where each position receives a weight of % dollar allocation. A portfolio of loans or savings products is no different in this regard. The weight for each loan-level or account-level observation would be % of balance for that product class at observation date (t):
w_i=B_i/(∑_(i=1)^N▒B_i )

The weights can be passed to a classification model as sample weight parameters. This effectively over-samples or under-samples each observation depending on whether they have a higher-than-average or lower-than-average balance, respectively.

## 4.	Classification and predicting class likelihood

Over-sampling or under-sampling is often used in machine learning for classification tasks on rare events data. The primary motivation is for refined calibration and performance. Often times classification accuracy alone will over-state the quality of a forecast when minority class occurrence is very rare (<1%). False positive and false negative rates, or Type I and II errors, provide a more nuanced view of majority and minority class performance. A model might achieve 99% accuracy but incorrectly classify every single minority outcome simply because 99%+ of observations are in the majority class. Re-sampling techniques artificially change the number of minority outcomes in data to improve false positive/false negative rates. The confusion matrix and senstivity/specificty rates are used to score the model on imbalanced data and adjust for .

In contrast, for financial cash flow analysis the model output is class likelihood, not class ID:
* Charge-off probability as credit risk adjustment or % balance outflow
* Prepayment probability as % balance outflow
* Attrition probability as % line and balance outflow 

For cash flow adjustments the class likelihood is used directly. Individual loans aren’t classified as default/non-defaulted, but instead a percentage of their balance is subtracted each month to adjust the portfolio valuation and cash flow. 

Re-sampling techniques introduce a bias that skews the average class probabilities. While for classification tasks this doesn’t significantly impact class prediction, since probabilities are rounded into 0/1 class IDs, the underlying class likelihoods will be misrepresentative of the baseline event rate, skewed in the direction of sampling. Over-sampling default indicators will lead to over-prediction of portfolio loss rates unless tied to balance amounts. Re-sampling should only be used when class ID is the ultimate variable of interest, not class likelihood.
