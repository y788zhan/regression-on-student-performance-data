library(ISLR)
library(leaps)
library(glmnet)
library(pls)

set.seed(1)

port = read.table('student-por.csv', sep = ';', header = TRUE)
port$G1 = NULL
port$G2 = NULL

## Attributes for both student-mat.csv (Math course) and student-por.csv (Portuguese language course) datasets: (not in particular order)
 # 1  school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira) 
 # 2  sex - student's sex (binary: 'F' - female or 'M' - male) 
 # 3  age - student's age (numeric: from 15 to 22) 
 # 4  address - student's home address type (binary: 'U' - urban or 'R' - rural) 
 # 5  famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3) 
 # 6  Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart) 
 # 7  Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education) 
 # 8  Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education) 
 # 9  Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') 
 # 10 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other') 
 # 11 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour) 
 # 12 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours) 
 # 13 failures - number of past class failures (numeric: n if 1<=n<3, else 4) 
 # 14 schoolsup - extra educational support (binary: yes or no) 
 # 15 famsup - family educational support (binary: yes or no) 
 # 16 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no) 
 # 17 activities - extra-curricular activities (binary: yes or no) 
 # 18 nursery - attended nursery school (binary: yes or no) 
 # 19 higher - wants to take higher education (binary: yes or no) 
 # 20 internet - Internet access at home (binary: yes or no) 
 # 21 romantic - with a romantic relationship (binary: yes or no) 
 # 22 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent) 
 # 23 freetime - free time after school (numeric: from 1 - very low to 5 - very high) 
 # 24 goout - going out with friends (numeric: from 1 - very low to 5 - very high) 
 # 25 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high) 
 # 26 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high) 
 # 27 health - current health status (numeric: from 1 - very bad to 5 - very good) 
 # 28 absences - number of school absences (numeric: from 0 to 93)
 # 29 guardian - student's guardian (nominal: 'mother', 'father' or 'other')
 # 30 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')

 # G3 - final grade
 ##

# convert all attributes to numeric
for (i in names(port)) {
	port[[i]] = as.numeric(port[[i]])
}

rows <- nrow(port)
cols <- ncol(port)

##################################################################
#                                                                #
# use cross validation to choose among models of different sizes #
#                                                                #
##################################################################

predict.regsubsets = function(object, newdata, id, ...) {
    form = as.formula(object$call[[2]])
    mat = model.matrix(form, newdata)
    coefi = coef(object, id = id)
    mat[, names(coefi)] %*% coefi
}


#########################################
#                                       #
#       Simple Multiple Regression      #
#                                       #
#########################################
reg.best = regsubsets(G3 ~ ., data = port, nvmax = cols)

# k-fold cross validation
k = 10 # 10 folds
folds = sample(1:k, rows, replace = TRUE)
cv.errors = matrix(NA, k, cols, dimnames = list(NULL, paste(1:cols)))

##
 # create matrix such that cv.errors[j, i]
 # is the test MSE for j-th cross-validation fold for the best i-variable model
 ##
for (j in 1:k) {
	best.fit = regsubsets(G3 ~ ., data = port[folds != j, ], nvmax = cols)
	for (i in 1:cols) {
		pred = predict(best.fit, port[folds == j, ], id = i) 
		cv.errors[j, i] = mean((port$G3[folds == j] - pred) ^ 2)
	}
}

mean.cv.errors = apply(cv.errors, 2, mean)
plot(mean.cv.errors, type = 'b', xlab = 'Number of Variables in Model',
	ylab = 'Mean MSE of Folds', main = 'CV on Simple Multiple Regression')

minMSE <- which.min(mean.cv.errors) # 8 (MSE = 7.729094)
coef(reg.best, minMSE)
##
 # (Intercept)      school        Fedu   studytime    failures   schoolsup      higher        Dalc      health 
 #  11.8199775  -1.4058998   0.2674100   0.5196096  -1.4401402  -1.2924009   1.8360104  -0.4315652  -0.2035014 
 ##


#############################################
#                                     		#
#              Ridge Regression             #
#                                     		#
#############################################
x = model.matrix(G3 ~ ., port)[, -1]
y = port$G3
# create lambda values from 1E-2 to 1E10
grid = 10 ^ seq(10, -2, length = 100)

train = sample(1:nrow(x), nrow(x) / 2)
test = (-train)
y.test = y[test]

# cross validation on lambda
ridge.mod = glmnet(x[train, ], y[train], alpha = 0, lambda = grid)
cv.out = cv.glmnet(x[train, ], y[train], alpha = 0)
bestridgelam = cv.out$lambda.min
ridge.pred = predict(ridge.mod, s = bestridgelam, newx = x[test, ])
mean((ridge.pred - y.test) ^ 2) # 7.236876

# build model using full dataset
out = glmnet(x, y, alpha = 0, lambda = grid)
ridge.coef = predict(out, type = 'coefficients', s = bestridgelam)
##
 # 31 x 1 sparse Matrix of class "dgCMatrix"
 #                        1
 # (Intercept) 10.493201874
 # school      -0.829508476
 # sex         -0.387208146
 # age          0.040154510
 # address      0.306390395
 # famsize      0.203340853
 # Pstatus      0.078218592
 # Medu         0.147416271
 # Fedu         0.159087041
 # Mjob         0.061974158
 # Fjob        -0.004003125
 # reason       0.062423888
 # guardian    -0.104138654
 # traveltime  -0.019912164
 # studytime    0.329543586
 # failures    -0.923940534
 # schoolsup   -0.762467085
 # famsup       0.047674481
 # paid        -0.393284045
 # activities   0.160307354
 # nursery     -0.102718775
 # higher       1.277642886
 # internet     0.292183367
 # romantic    -0.256328049
 # famrel       0.083937382
 # freetime    -0.116442203
 # goout       -0.043645496
 # Dalc        -0.207603955
 # Walc        -0.108940461
 # health      -0.113486235
 # absences    -0.024212624
 ##

#######################################
#                                     #
#              The LASSO              #
#                                     #
#######################################
# cross validation on lambda
lasso.mod = glmnet(x[train, ], y[train], alpha = 1, lambda = grid)
cv.out = cv.glmnet(x[train, ], y[train], alpha = 1)
plot(cv.out)
bestlassolam = cv.out$lambda.min
lasso.pred = predict(lasso.mod, s = bestlassolam, newx = x[test, ])
mean((lasso.pred - y.test) ^ 2) # 8.224117

# build model using full dataset
out = glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef = predict(out, type = 'coefficients', s = bestlassolam)
##
 # 31 x 1 sparse Matrix of class "dgCMatrix"
 #                         1
 # (Intercept) 10.7655459456
 # school      -1.2255598684
 # sex         -0.4002936925
 # age          0.0655700419
 # address      0.1882049895
 # famsize      0.1261151205
 # Pstatus      .           
 # Medu         0.1280740176
 # Fedu         0.1519460162
 # Mjob         0.0065880246
 # Fjob         .           
 # reason       0.0004234013
 # guardian    -0.0086731061
 # traveltime   .           
 # studytime    0.3836917731
 # failures    -1.3391585875
 # schoolsup   -1.0526553941
 # famsup       .           
 # paid        -0.2285035341
 # activities   0.0765949986
 # nursery     -0.0009631168
 # higher       1.6427771104
 # internet     0.2144119363
 # romantic    -0.2428805871
 # famrel       0.0349788572
 # freetime    -0.0744927260
 # goout       -0.0001744240
 # Dalc        -0.2248018713
 # Walc        -0.0987260532
 # health      -0.1312875122
 # absences    -0.0222306602
 ##


#################################################
#                                               #
#         Principal Component Regression        #
#                                               #
#################################################
# cross validation on number of components
pcr.fit = pcr(G3 ~ ., data = port, subset = train, scale = TRUE, validation = 'CV')
summary(pcr.fit) # minimum validation error occurs at comp = 19
validationplot(pcr.fit, val.type = 'MSEP')
# note comp = 15 has a very similar validation error to comp = 19, but gives us more dimension reduction

pcr.pred = predict(pcr.fit, x[test, ], ncomp = 15) # 7.212316
mean((pcr.pred - y.test) ^ 2)

# build model using full dataset
pcr.fit = pcr(G3 ~ ., data = port, scale = TRUE, ncomp = 15)
summary(pcr.fit)
##
 # Data: 	X dimension: 649 30 
 #          Y dimension: 649 1
 # Fit method: svdpc
 # Number of components considered: 15
 # TRAINING: % variance explained
 #     1 comps  2 comps  3 comps  4 comps  5 comps  6 comps  7 comps  8 comps  9 comps  10 comps  11 comps
 # X     10.31    18.65    24.54    29.31    33.72    38.06    42.29    46.44    50.01     53.45     56.84
 # G3    19.13    22.68    22.69    22.85    24.92    25.78    25.80    26.35    26.38     26.48     27.72
 #     12 comps  13 comps  14 comps  15 comps
 # X      60.11     63.24     66.28     69.26
 # G3     28.33     29.11     29.11     31.65
 ##


#########################################
#                                       #
#         Partial Least Squares         #
#                                       #
#########################################
# cross validation on number of components
pls.fit = plsr(G3 ~ ., data = port, subset = train, scale = TRUE, validation = 'CV')
summary(pls.fit) # minimum validation error occurs at comp = 2
validationplot(pls.fit, val.type = 'MSEP')

pls.pred = predict(pls.fit, x[test, ], ncomp = 2)
mean((pls.pred - y.test) ^ 2) # 8.413339

# build model using full dataset
pls.fit = plsr(G3 ~ ., data = port, scale = TRUE, ncomp = 2)
summary(pls.fit)
##
 # Data: X dimension: 649 30 
 # 	     Y dimension: 649 1
 # Fit method: kernelpls
 # Number of components considered: 2
 # TRAINING: % variance explained
 #              1 comps   2 comps
 #          X     9.736     14.36
 #          G3   29.150     33.80
 ##


##############################
#                            #
#		Bootstrapping        #
#                            #
##############################

##
 # function to perform bootstrapping on model
 # allow to specify sampleSize of each bootstrap, and number of samples to take
 # assumes port data already exists
 ##
getResampleMSE = function(method, modelObject, specific, sampleSize, numSamples) {
	BSErrors <- vector(, numSamples)

	for (i in 1:numSamples) {
		resample = port[sample(1:rows, sampleSize, replace = TRUE), ]
		resample.y = resample$G3
		resample.x = model.matrix(G3 ~ ., resample)[, -1]

		if (method == 'simple') {
			resample.simpPred = predict(modelObject, resample, id = specific)
			BSErrors[i] = mean((resample.simpPred - resample.y) ^ 2)
		} else if (method == 'lasso' || method == 'ridge') {
			resample.lassoPred = predict(modelObject, s = specific, newx = resample.x)
			BSErrors[i] = mean((resample.lassoPred - resample.y) ^ 2)
		} else if (method == 'pls' || method == 'pcr') {
			resample.PLSPred = predict(modelObject, resample.x, ncomp = specific)
			BSErrors[i] = mean((resample.PLSPred - resample.y) ^ 2)
		} else {
			break
		}
	}

	hist(BSErrors, main = paste(method, 'bootstrapping', sep = ' '))
	mean(BSErrors)
}

# reseed
set.seed(1)

sampleSize <- 100
numSamples <- 1000

# simple multiple regression
getResampleMSE('simple', reg.best, minMSE, sampleSize, numSamples) # 7.121684

# the lasso
getResampleMSE('lasso', lasso.mod, bestlassolam, sampleSize, numSamples) # 7.03579

# ridge regression
getResampleMSE('ridge', ridge.mod, bestridgelam, sampleSize, numSamples) # 7.251852

# PCR
getResampleMSE('pcr', pcr.fit, 15, sampleSize, numSamples) # 7.066262

# PLS
getResampleMSE('pls', pls.fit, 2, sampleSize, numSamples) # 6.87703


#########################
#                       #
#		Conclusion		#
#                       #
#########################

summary(port$G3)
##
 #  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
 #  0.00   10.00   12.00   11.91   14.00   19.00
 ##

##
 # Performance Summary:
 #
 # simple multiple regression performance:
 # bootstrapped MSE = 7.121684
 # RMSE = 2.6686
 # RMSE is 14.05% of G3's range
 #
 # the ridge regression's performance:
 # bootstrapped MSE = 7.251852
 # RMSE = 2.6929
 # RMSE is 14.17% of G3's range
 #
 # the lasso's performance:
 # bootstrapped MSE = 7.03579
 # RMSE = 2.6525
 # RMSE is 13.96% of G3's range
 #
 # PCR's performance:
 # bootstrapped MSE = 7.066262
 # RMSE = 2.6582
 # RMSE is 13.99% of G3's range
 #
 # PLS's performance:
 # bootstrapped MSE = 6.87703
 # RMSE = 2.6224
 # RMSE is 13.80% of G3's range
 #
 #
 # Interpretation:
 # 
 # Multiple linear regression provided the most subset-selection, with only 8 predictors selected. 
 # It seemed to agree mostly with the only other subset-selection method, LASSO regression. 
 # All predictors selected by cross-validation on regsubsets have a coefficient of at least 0.13 in absolute value 
 # in LASSO regression. Both methods placed their highest coefficients on predictors such as school, failures, 
 # schoolsup and higher.
 #
 # Multiple linear regression performed relatively well with resampled data.
 #
 # Comparing LASSO and ridge regression, ridge regression seemed to perform much better than LASSO with the test data.
 # Note that the test data here is obtained simply through selecting a random half of the original data.
 # Ridge regression's better performance may be attributed to the fact that it did not perform subset selection, and
 # therefore had lower bias. But when compared in bootstrapped RMSE, the LASSO performed better. This could be an
 # indication that ridge regression overfitted the dataset.
 #
 # Similar results came from PCR and PLS. PLS provided significanlty more dimension reduction than PCR, and in turn
 # peformance worse on the (non-boostrapped) test data, but performed better on the boostrapped test data. This suggests
 # that having 15 components in PCR may have led to overfitting and excessive variance.
 #
 # The LASSO regression eliminated 4 predictors (Pstatus, Fjob, traveltime, famsup), and came very close to elminating 5
 # other predictors (Mjob, guardian, reason, nursery, goout). Given 30 predictors and only ~600 observations, it is very
 # difficult to determine which predictors truly affect the response and which coefficients are correctly assigned.
 # From the models and using common sense, we can likely agree that predictors such as the ones eliminated by LASSO are
 # just noise. And predictors such as school, failures, schoolsup, higher and study time are more strongly correlated
 # with student performance.
 # 
 ##
