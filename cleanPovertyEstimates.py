import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



#####################################
# Cleaning Poverty Estimate Dataset #
#####################################

dfPovertyEstimate = pd.read_csv('datasets/PovertyEstimates_2020.csv', usecols=['FIPS_code', 'Stabr', 'Area_name','POVALL_2020', 'PCTPOVALL_2020', 'POV017_2020', 'PCTPOV017_2020','POV517_2020', 'PCTPOV517_2020', 'MEDHHINC_2020'])
dfDeaths = pd.read_csv('datasets/covid_deaths_usafacts.csv')
dfPovertyEstimate = dfPovertyEstimate.replace(',','', regex=True)
dfPovertyEstimate['POVALL_2020'] = pd.to_numeric(dfPovertyEstimate['POVALL_2020'])
dfPovertyEstimate['MEDHHINC_2020'] = pd.to_numeric(dfPovertyEstimate['MEDHHINC_2020'])

#######################################
# Merging Poverty Estimate with Death #
#######################################

col_to_compare_PovEst = 'FIPS_code'
col_to_compare_Deaths = 'countyFIPS'

dfPovertyEstimate.rename(columns={col_to_compare_PovEst: 'FIPS_code'}, inplace=True)
dfDeaths.rename(columns={col_to_compare_Deaths: 'FIPS_code'}, inplace=True)

merged_df = pd.merge(dfPovertyEstimate, dfDeaths, left_on='FIPS_code', right_on='FIPS_code', how='inner')
merged_df_2 = merged_df.copy()
columns_to_delete = ['FIPS_code', 'StateFIPS','PCTPOVALL_2020', 'POV017_2020','PCTPOV017_2020','POV517_2020','PCTPOV517_2020','MEDHHINC_2020']
merged_df.drop(columns_to_delete, axis=1, inplace=True)
columns_to_delete = ['FIPS_code', 'StateFIPS','PCTPOVALL_2020', 'POV017_2020','PCTPOV017_2020','POV517_2020','PCTPOV517_2020','POVALL_2020']
merged_df_2.drop(columns_to_delete, axis=1, inplace=True)


#######################################
# Calculating Total Deaths per County #
#######################################

merged_df['total_dead'] = merged_df.sum(axis=1)

merged_df.dropna(inplace=True)
merged_df_2['total_dead'] = merged_df.sum(axis=1)

merged_df_2.dropna(inplace=True)

#######################################################
# Calculate Linear Regression POVALL_2020 and Display #
#######################################################
merged_df.drop(merged_df[merged_df['total_dead'] >= 840289.0].index, inplace=True)
X = merged_df[['POVALL_2020']]
y = merged_df['total_dead']


X_train, X_test, y_train, y_test = train_test_split(merged_df[['POVALL_2020']], merged_df['total_dead'], test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", mse)

# plt.scatter(X_train, y_train)
# plt.scatter(X_test,y_test)
# plt.plot(X_train, model.predict(X_train))
# plt.show()

model = LinearRegression()
model.fit(X, y)


print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)
print("R-squared: ", model.score(X, y))


plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Estimate of people of all ages in poverty 2020')
plt.ylabel('Total Deaths per County in 2020')
plt.title('Linear Regression of Deaths based on 2020 Poverty Estimates')
plt.show()

################################################
# Calculate Linear Regression MHHI and Display #
################################################
merged_df_2.drop(merged_df_2[merged_df_2['total_dead'] >= 840289.0].index, inplace=True)
merged_df_2.head(15)
X = merged_df_2[['MEDHHINC_2020']]
y = merged_df_2['total_dead']
print(merged_df_2.describe())

X_train, X_test, y_train, y_test = train_test_split(merged_df_2[['MEDHHINC_2020']], merged_df_2['total_dead'], test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Mean Squared Error: ", mse)
print("Root Mean Squared Error: ", mse)



model = LinearRegression()
model.fit(X, y)


print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)
print("R-squared: ", model.score(X, y))


plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Estimate of median household income 2020')
plt.ylabel('Total Deaths per County in 2020')
plt.title('Linear Regression of Deaths based on 2020 Median Household Income')
plt.show()


