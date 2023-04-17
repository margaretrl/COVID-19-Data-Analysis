import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

#####################################
#Load the dataset and merge datasets#
#####################################

# Load the datasets
dfVaccination = pd.read_csv('/Users/marijatravoric/IDC 4140/COVID-19-Data-Analysis/datasets/Vaccinations/updated_vaccinations.csv')
dfDeath = pd.read_csv('/Users/marijatravoric/IDC 4140/COVID-19-Data-Analysis/datasets/covid_deaths_usafacts.csv')

dfDeath = dfDeath.drop(['County Name', 'countyFIPS', 'StateFIPS'], axis = 1)
dfDeath = pd.melt(dfDeath, id_vars=['State'], var_name = 'date', value_name = 'cases')

# Merge the two datasets based on matching date and location columns
df = pd.merge(dfVaccination, dfDeath, left_on=['date', 'location'], right_on=['date', 'State'])

# Drop the extra date and state columns
df = df.drop(['location'], axis=1)
df = df.rename(columns={'date': 'Date', 'cases': 'Total Deaths'})

# Group the data by state and date and sum up the number of deaths and vaccinations for each group
df = df.groupby(['State', 'Date'])['Total Deaths', 'people_fully_vaccinated'].sum().reset_index()


######################################
# Calculating Total Deaths per State #
######################################

# Group the data by state and date and sum up the number of deaths and vaccinations for each group
df = df.groupby(['State', 'Date'])['Total Deaths', 'people_fully_vaccinated'].sum().reset_index()

###########################################
# Calculate Linear Regression and Display #
###########################################
#perform linear regression
X = df[['people_fully_vaccinated']]
y = df['Total Deaths']
model = LinearRegression()
model.fit(X, y)

# plot the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('People Fully Vaccinated by State')
plt.ylabel('Total Deaths by State')
plt.title('Linear Regression of Total Vaccinations vs COVID-19 Deaths by State')
plt.show()



















# #Check column names in the merged dataset
# print(df.columns)

# # Check for missing values
# print(df.head())