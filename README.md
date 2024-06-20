# HOUSE PRICE PREDICTION (KING COUNTY, WA)

## Overview 
In this project a linear regression analysis on the [Kings County Housing Data](https://github.com/Amberlynnyandow/dsc-1-final-project-online-ds-ft-021119/tree/master/kc_house_data.csv)
is performed. The goal is to explore the dataset and generate a model to predict the price of Houses in Kings 
County in Washington State. This analysis and the results that follow is of importance to homeowners, realtors 
and other stakeholders in the King County Housing Market. 

## Model Assumptions
The linear regression model operates under the following assumptions:
- Regression residuals must be normaly distributed
- The dependent variable and the independent variable have a linear relationship.
- The residuals are homoskedastic and not autocorrelated. 
- Absence of Multicollinearity bewtween the indeppendent variables.
To read more on the assumptions of linear regression, click [here](https://www.statisticssolutions.com/assumptions-of-linear-regression/)

## Data Preprocessing
Prior to model building, the dataset must be cleaned and preprocessed. This prepares the dataset to ensure
it goes into the model, unbiased and in its best form. Dealing with missing values, datatype transformation,
categorical encoding, outlier detection and multicollinearity check were the main preprocessing steps carried
out on the data.

### Missing Values and Datatype transformation
The first step after importing the dataset was to ensure that all variables, whether feature or target, conformed
to the right datatype and had no missing elements. only four features in the dataset had missing observation and 
this was dealt with by assigning modal values to fill in the missing values in the categorical features and assigning
mean values to fill in the missing values in the numeric features. To achieve correct datatypes, four variables - 
condition, grade, zipcode and waterfront were transformed into categorical variables according to the [data description](support/data-description.md)
for the dataset.

### Categorical Encoding
To ensure all features exist as numeric values for input into the model, the four categorical variables in the dataset - 
condition, grade, zipcode and waterfront were encoded into numbers using Target encoding. Target encoding was chosen over
other methods such as one-hot encoding or ordinal encoding due to the high number of categories in some of the features to
be encoded.

### Outlier Detection
To deal with outliers that exist in the dataset, outliers were removed by cutting off all values that exist above or
below three standard deviations from the mean values of each feature. After performing this operation, 15% of the total 
dataset was dropped from the analysis. This helps to safeguard the model from values that cous potentially swing the 
model during training. 

### Multicollinearity Check
A multicollinearity check is also performed on the model to ensure the coefficients are not biased. The precision of
regression coefficients or regression predictions may be decreased if highly correlated explanatory variables are
included in the model. Multicollinearity which can be detected by high variance inflation factors (VIF) values. To deal 
with multicollinearity, data features with a VIF above 5 were filtered out from the dataset.Multicollinearity renders 
regression coefficient estimates unreliable and the standard errors of the slope coefficients become artificially 
inflated, leading to problems with the statistical significance of the regression coefficients.

### Dropped Columns
After all data preprocessing was performed, several columns were dropped from the datset and the final dataset used to
develop the price prediction model had 18603 observations with 14 features. The final fatures used in the house price
prediction model were bedrooms, bathrooms, sqft_lot, floors, waterfront, view, condition, grade, yr_built, yr_renovated,
zipcode, sqft_living15, and sqft_lot15.

## Data Visualization


## Price Prediction Model

**Model Building**
- The test-train-split was used on the dataset to test the predictive ability of the model. This helps
to check how the model performs on data not passed through it, similar to how it is expected to perform
in the real world.

- The 80:20 test-train split was used for training and testing this model.

- The linear regression model was instantiated and fitted/trained on the training dataset.

- Predictions on the test dataset were made using the trained model and the model summary was extracted. The model summary
displays the model's intercept and coefficients, along with accompanying hypothesis tests.

**Model Results and Hypothesis Testing**
- The model had a constant value of $1,252,000 (One million, two hundred and fifty thousand dollars). This
represents the average house price in King County when no additional features are added, i.e.,  the house 
price if all houses were the same.

- The coefficients represent the slopes. In this model, the slopes represent the change in house prices
caused by a unit change in one feature while holding all other features constant. For example, the 
coefficient for the year the house was built is -1711, this implies that for every year the house gets older,
the price drops by $1,711 while holding all other variables constant. In the same vein, an additional bedroom
causes the house price to rise by $17,990.

- The model had an R-squared value of 0.771, implying that the model was able to explain about 77.1% of 
the variation in the dependent variable (i.e., King County house prices).

- The model had an F-statistic of 4802. This implies that jointly, the model features do have a significant
effect on the dependent variable (i.e., King County house prices). The null hypothesis of non-significance
was rejected at 1% level of significance. 

- All model features house renovation year' have large t-values. This implies that while holding all other
 features constant, each model feature except for the 'house renovation year' has a significant effect on 
 King County house prices. For every feature except the 'house renovation year', the null hypothesis of 
non-significance was rejected at 1% level of significance.

## Model Performance and Diagnostics
### Model Performance
The measure of the error in the  model gives a good sense of its performance. Three error types are examined in this project. The errors are calculated based on the model performance on the test
dataset after the test-train split. The errors computed are:

- Mean Absolute Error (MAE): which is the mean of the absolute value of errors - $73,547
- Mean Squared Error (MSE): which is the mean of the squared errors - $10,823,457,615
- Root Mean Squared Error (RMSE): which is the square root of the mean of the squared errors - $104,035

### Model Diagnostics

- **Heteroscedasticity** : The Breusch-Pagan or White's test can be used to check for Homoskedasticity in the model.
The null hypothesis states that the variances for the errors are equal while the alternative hypothesis is that the 
variances are not equal (or at least one is different from the others). Failure to reject the null hypothesis, 
indicated by the presence of a high p-value implies that the model produces homoskedastic errors. The Breusch-Pagan
test was used and the results failed to reject the null hypothesis, indicating that the model was homoskedastic.

- **Autocorrelation** : Autocorrelation refers to the degree of correlation between the values of the same variables
across different observations in the data. The null hypothesis states that the errors are random (non-autocorrelation)
while the alternative hypothesis states that the errors are not random (autocorrelation). A failure to reject the null
hypothesis, indicated by the presence of a high p-value implies that the model features are non-autocorrelated. 
Alternatively, the autocorrelation function plot could be used. The ACF plot clearly shows that the model does not suffer
from autocorrelation in its residuals as every lag beyond the first lag is very close to zero.

- **Normality** : The assumption of normality is usually checked using the QQ-plot visualization. Normally distributed 
errors usually cluster around the diagonal of a plot. Alternatively, checking that the mean of the residual is 0 is a 
way of checking for the normality of a model's residuals.
The model QQ-Plot mostly follows the diagonal line in the plot and mean of residuals are very close to zero, indicating 
that the model errors are normally distributed. 

## Summary
In this study, a house price prediction model was developed for King County, WA. Prior to model building, missing values
were dealt with, datatype transformation was carried out, categorical features were encoded, outliers were detected and 
removed from the data. Finally, multicollinearity checks were carried out on the data to ensure highly correlated features
were not introduced into the model.

The model performed decently, explaining 77% of the price using 14 features, attaining a high degree of accuracy in predicting
house prices. It did not violate the statistical assumptions under which it was developed and overall, the model predictions 
were off cumulatively by $73,547. An improvement on other models developed using the King County Housing Data.

In conclusion, the house price prediction model would prove useful to real estate stakeholders in King County, WA, offering
them precise and actionable insights. With further refinement and addition of new features, this model has the potential to
greatly assist in investment decisions, market analysis, and strategic planning in the real estate sector.



