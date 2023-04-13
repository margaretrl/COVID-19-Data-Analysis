import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

dfPovertyEstimate = pd.read_csv('datasets/PovertyEstimates_2020.csv', usecols=['FIPS_code', 'Stabr', 'Area_name','POVALL_2020', 'PCTPOVALL_2020', 'POV017_2020', 'PCTPOV017_2020','POV517_2020', 'PCTPOV517_2020', 'MEDHHINC_2020'])

dfDeaths = pd.read_csv('datasets/covid_deaths_usafacts.csv')

dfPovertyEstimate = dfPovertyEstimate.replace(',','', regex=True)
dfPovertyEstimate['POVALL_2020'] = pd.to_numeric(dfPovertyEstimate['POVALL_2020'])

# #####################################
# # Cleaning Poverty Estimate Dataset #
# #####################################

# Merge the two dataframes using a common row
#Select the columns to compare and rename them to the same name
col_to_compare_PovEst = 'FIPS_code'
col_to_compare_Deaths = 'countyFIPS'
dfPovertyEstimate.rename(columns={col_to_compare_PovEst: 'FIPS_code'}, inplace=True)
dfDeaths.rename(columns={col_to_compare_Deaths: 'FIPS_code'}, inplace=True)

# Merge the two datasets on the selected column
merged_df = pd.merge(dfPovertyEstimate, dfDeaths, left_on='FIPS_code', right_on='FIPS_code', how='inner')
columns_to_delete = ['FIPS_code', 'StateFIPS','PCTPOVALL_2020', 'POV017_2020','PCTPOV017_2020','POV517_2020','PCTPOV517_2020','MEDHHINC_2020']
merged_df.drop(columns_to_delete, axis=1, inplace=True)


merged_df['total_dead'] = merged_df.sum(axis=1)

# Check the merged dataframe for missing data and clean as necessary
merged_df.dropna(inplace=True)



# Split the merged dataframe into independent and dependent variables
X = merged_df[['POVALL_2020']]
y = merged_df['total_dead']

# Create a linear regression model using scikit-learn
model = LinearRegression()
model.fit(X, y)

# Print the regression coefficients and other statistics
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)
print("R-squared: ", model.score(X, y))

# Visualize the results using a scatter plot with the regression line overlaid
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.title('Linear Regression Analysis')
plt.show()




# dfPovertyEstimate = dfPovertyEstimate.fillna(0)
# dfPovertyEstimate = dfPovertyEstimate.replace(',','', regex=True)
# dfPovertyEstimate['POVALL_2020'] = pd.to_numeric(dfPovertyEstimate['POVALL_2020'])


# #########################
# # Keeping all 2020 info #
# #########################

# # Select columns to keep
# columns_to_keep = ['countyFIPS', 'County Name', 'State', 'StateFIPS']

# # Select columns to delete that don't start with 2020
# columns_to_delete = dfDeaths.filter(regex='^(?!2020).').columns

# # Exclude specific columns from the list of columns to delete
# columns_to_delete = [col for col in columns_to_delete if col not in columns_to_keep]

# # Drop the columns to delete from the dataset
# dfDeaths.drop(columns_to_delete, axis=1, inplace=True)

# #check for null values
# num_null_values = dfDeaths.isnull().sum().sum()

# if num_null_values == 0:
#     print("The updated dataset does not contain any null values.")
# else:
#     print(f"The updated dataset contains {num_null_values} null values.")

# # Save the updated dataset to a new file
# dfDeaths.to_csv('datasets/updated_Deaths_dataset.csv', index=False)


# #####################################################
# # Checking how many rows match between each dataset #
# #####################################################

# #Select the columns to compare and rename them to the same name
# col_to_compare_PovEst = 'FIPS_code'
# col_to_compare_Deaths = 'countyFIPS'
# dfPovertyEstimate.rename(columns={col_to_compare_PovEst: 'column_name'}, inplace=True)
# dfDeaths.rename(columns={col_to_compare_Deaths: 'column_name'}, inplace=True)

# # Merge the two datasets on the selected column
# merged_df = pd.merge(dfPovertyEstimate, dfDeaths, left_on='column_name', right_on='column_name', how='inner')

# # Count how many rows match
# num_matches = merged_df.shape[0]

# # Print the result
# print(f"There are {num_matches} rows that match on the '{col_to_compare_PovEst}' column in dataset 1 and '{col_to_compare_Deaths}' column in dataset 2.")


# ##############################
# # Creating Linear Regression #
# ##############################

# columns_to_delete = ['column_name', 'StateFIPS','PCTPOVALL_2020', 'POV017_2020','PCTPOV017_2020','POV517_2020','PCTPOV517_2020','MEDHHINC_2020']
# edit_merged_df =merged_df.copy()
# edit_merged_df.drop(columns_to_delete, axis=1, inplace=True)


# edit_merged_df['total_dead'] = edit_merged_df.sum(axis=1)
# # print(edit_merged_df.dtypes)
# # print(merged_df['column_name'].dtypes)
# print(edit_merged_df.describe()[['POVALL_2020', 'total_dead']])




# # Box Plot
# import seaborn as sns
# # sns.boxplot(edit_merged_df['POVALL_2020'])
# pov = np.where(edit_merged_df['POVALL_2020']>1290000 )
# dead = np.where(edit_merged_df['total_dead']>2700000)

# print(dead)
# povDeath = sorted(list(pov[0])+list(dead[0]))

# edit_merged_df.drop(povDeath,inplace=True)


# # fig, ax = plt.subplots(figsize = (7,7))

# # ax.scatter(edit_merged_df['POVALL_2020'], edit_merged_df['total_dead'])
 
# # # x-axis label
# # ax.set_xlabel('Poverty Estimate')
 
# # # y-axis label
# # ax.set_ylabel('Total Dead')
# # plt.show()
# # plt.show()
# # X_train, X_test, y_train, y_test = train_test_split( edit_merged_df['POVALL_2020'], edit_merged_df['total_dead'], test_size=0.3, random_state=42)


# model = LinearRegression()
# model.fit(edit_merged_df['POVALL_2020'], y_train)

# plt.scatter(edit_merged_df['POVALL_2020'], y_train)
# plt.scatter(X_test,y_test)
# plt.plot(X_train, model.predict(X_train.values.reshape(-1,1)))

# plt.show()