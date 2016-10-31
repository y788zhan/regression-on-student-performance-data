# Regression on Student Performance Data

[Data source: Universit of California, Irvine, Machine Learning respository](http://archive.ics.uci.edu/ml/datasets/Student+Performance)

## Performance Summary:

| Model                          | Bootstrapped MSE | Root MSE | RMSE as % of Response Range |
| ------------------------------ |:----------------:|:--------:|:---------------------------:|
| Multiple Linear Regression     | 7.121684         | 2.6686   | 14.05                       |
| Ridge Regression               | 7.251852         | 2.6929   | 14.17                       |
| LASSO Regression               | 7.03579          | 2.6525   | 13.96                       |
| Principal Component Regression | 7.066262         | 2.6582   | 13.99                       |
| Partial Least Squares          | 6.87703          | 2.6224   | 13.80                       |

## Interpretation:

Multiple linear regression provided the most subset-selection, with only 8 predictors selected. 
It seemed to agree mostly with the only other subset-selection method, LASSO regression. 
All predictors selected by cross-validation on regsubsets have a coefficient of at least 0.13 in absolute value 
in LASSO regression. Both methods placed their highest coefficients on predictors such as `school`, `failures`, 
`schoolsup` and `higher`.

Multiple linear regression performed relatively well with resampled data.
Comparing LASSO and ridge regression, ridge regression seemed to perform much better than LASSO with the test data.
Note that the test data here is obtained simply through selecting a random half of the original data.
Ridge regression's better performance may be attributed to the fact that it did not perform subset selection, and
therefore had lower bias. But when compared in bootstrapped RMSE, the LASSO performed better. This could be an
indication that ridge regression overfitted the dataset.

Similar results came from PCR and PLS. PLS provided significanlty more dimension reduction than PCR, and in turn
peformance worse on the (non-boostrapped) test data, but performed better on the boostrapped test data. This suggests
that having 15 components in PCR may have led to overfitting and excessive variance.

The LASSO regression eliminated 4 predictors (`Pstatus`, `Fjob`, `traveltime`, `famsup`), and came very close to
elminating 5 other predictors (`Mjob`, `guardian`, `reason`, `nursery`, `goout`). Given 30 predictors and only ~600
observations, it is very difficult to determine which predictors truly affect the response and which coefficients are
correctly assigned.
From the models and using common sense, we can likely agree that predictors such as the ones eliminated by LASSO are
just noise. And predictors such as `school`, `failures`, `schoolsup`, `higher`, and `studytime` are more strongly
correlated with student performance.