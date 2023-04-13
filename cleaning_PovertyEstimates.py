import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#####################################
# Cleaning Poverty Estimate Dataset #
#####################################

dfPovertyEstimate = pd.read_csv('datasets/PovertyEstimates_2020.csv', usecols=['FIPS_code', 'Stabr', 'Area_name','POVALL_2020', 'PCTPOVALL_2020', 'POV017_2020', 'PCTPOV017_2020','POV517_2020', 'PCTPOV517_2020', 'MEDHHINC_2020'])
dfDeaths = pd.read_csv('datasets/covid_deaths_usafacts.csv')
dfPovertyEstimate = dfPovertyEstimate.replace(',','', regex=True)
dfPovertyEstimate['POVALL_2020'] = pd.to_numeric(dfPovertyEstimate['POVALL_2020'])

#######################################
# Merging Poverty Estimate with Death #
#######################################

col_to_compare_PovEst = 'FIPS_code'
col_to_compare_Deaths = 'countyFIPS'

dfPovertyEstimate.rename(columns={col_to_compare_PovEst: 'FIPS_code'}, inplace=True)
dfDeaths.rename(columns={col_to_compare_Deaths: 'FIPS_code'}, inplace=True)

merged_df = pd.merge(dfPovertyEstimate, dfDeaths, left_on='FIPS_code', right_on='FIPS_code', how='inner')
columns_to_delete = ['FIPS_code', 'StateFIPS','PCTPOVALL_2020', 'POV017_2020','PCTPOV017_2020','POV517_2020','PCTPOV517_2020','MEDHHINC_2020']
merged_df.drop(columns_to_delete, axis=1, inplace=True)


#######################################
# Calculating Total Deaths per County #
#######################################

merged_df['total_dead'] = merged_df.sum(axis=1)

merged_df.dropna(inplace=True)


###########################################
# Calculate Linear Regression and Display #
###########################################
X = merged_df[['POVALL_2020']]
y = merged_df['total_dead']


model = LinearRegression()
model.fit(X, y)


print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)
print("R-squared: ", model.score(X, y))


plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.title('Linear Regression Analysis')
plt.show()
