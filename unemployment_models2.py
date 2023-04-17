import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import math


#####################################
# Cleaning Poverty Estimate Dataset #
#####################################

dfUnemployment = pd.read_csv(
    "/Users/margaretrivas/Desktop/IDC4140/finalProject/clean_Unemployment.csv"
)
dfDeaths = pd.read_csv("datasets/covid_deaths_usafacts.csv")

#######################################
# Merging Poverty Estimate with Death #
#######################################

col_to_compare_UnemRate = "FIPS_code"
col_to_compare_Deaths = "countyFIPS"

dfUnemployment.rename(columns={col_to_compare_UnemRate: "FIPS_code"}, inplace=True)
dfDeaths.rename(columns={col_to_compare_Deaths: "countyFIPS"}, inplace=True)

merged_df = pd.merge(
    dfUnemployment, dfDeaths, left_on="FIPS_code", right_on="countyFIPS", how="inner"
)
merged_df_2 = merged_df.copy()


#######################################
# Calculating Total Deaths per County #
#######################################

merged_df["total_dead"] = merged_df.sum(axis=1)

merged_df.dropna(inplace=True)
merged_df_2["total_dead"] = merged_df.sum(axis=1)

merged_df_2.dropna(inplace=True)

##########################################################
# Calculate Linear Regression Unemployment Rates/Display #
##########################################################
merged_df.drop(merged_df[merged_df["total_dead"] >= 840289.0].index, inplace=True)

low_limit, high_limit = merged_df["Unemployment_rate_2020"].quantile([0.005, 0.85])
merged_df = merged_df[
    (merged_df["Unemployment_rate_2020"] >= low_limit)
    & (merged_df["Unemployment_rate_2020"] <= high_limit)
]

X = merged_df[["Unemployment_rate_2020"]]
y = merged_df["total_dead"]


model = LinearRegression()
model.fit(X, y)


print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)
print("R-squared: ", model.score(X, y))


plt.scatter(X, y)
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Unemployment Rates in 2020")
plt.ylabel("Total Deaths per County in 2020")
plt.title("Linear Regression of Deaths based on 2020 Unemployment Rates")
plt.show()

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
se = math.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", se)


################################################
# Calculate Linear Regression MHHI and Display #
################################################

# merged_df_2.drop(merged_df_2[merged_df_2["total_dead"] >= 840289.0].index, inplace=True)
# merged_df_2["Median_Household_Income_2020"] = (
#     merged_df_2["Median_Household_Income_2020"].str.replace(",", "").astype(float)
# )
# merged_df_2.drop(
#     merged_df_2[merged_df_2["Median_Household_Income_2020"] >= 120000].index,
#     inplace=True,
# )

# merged_df_2["Median_Household_Income_2020"] = merged_df_2[
#     "Median_Household_Income_2020"
# ].astype(str)

# merged_df_2.head(15)
# X = (
#     merged_df_2["Median_Household_Income_2020"]
#     .str.replace(",", "")
#     .astype(float)
#     .values.reshape(-1, 1)
# )
# y = merged_df_2["total_dead"]
# print(merged_df_2.describe())


# model = LinearRegression()
# model.fit(X, y)


# print("Coefficients: ", model.coef_)
# print("Intercept: ", model.intercept_)
# print("R-squared: ", model.score(X, y))


# plt.scatter(X, y)
# plt.plot(X, model.predict(X), color="red")
# plt.xlabel("Estimate of median household income 2020")
# plt.ylabel("Total Deaths per County in 2020")
# plt.title("Linear Regression of Deaths based on 2020 Median Household Income")
# plt.show()

## BAR GRAPH ##

## Explanations

# The intercept value represents the predicted mean response of the dependent variable (total deaths per county in 2020) when the independent variable (unemployment rate in 2020) is zero. So, according to the linear regression model, if the unemployment rate in a county were zero, the model predicts that the average number of deaths per county would be around 38742.

# The R-squared value represents the proportion of the variance in the dependent variable (total deaths per county in 2020) that can be explained by the independent variable (unemployment rate in 2020). An R-squared value of 0.11 indicates that only 11% of the variance in the dependent variable can be explained by the independent variable.

# The Mean Squared Error (MSE) is a measure of the average squared difference between the actual and predicted values of the dependent variable. In this case, the MSE value of 23845910190.20 indicates that, on average, the predicted total deaths per county in 2020 are off by about 23845910190.20.

# The Root Mean Squared Error (RMSE) is the square root of the MSE and it represents the standard deviation of the residuals (differences between the actual and predicted values). In this case, the RMSE value of 154421.21 indicates that the standard deviation of the residuals is around 154421.21.
