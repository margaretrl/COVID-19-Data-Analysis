import pandas as pd
import numpy as np

dfPop = pd.read_csv('/Users/marijatravoric/IDC 4140/COVID-19-Data-Analysis/datasets/covid_county_population_usafacts.csv')

dfDeaths = pd.read_csv('/Users/marijatravoric/IDC 4140/COVID-19-Data-Analysis/datasets/deaths_by_year.csv')


# #####################################################
# # Checking how many rows match between each dataset #
# #####################################################
# #There are 5743 rows that match

# #Select the columns to compare and rename them to the same name
# col_to_compare_Pop = 'countyFIPS'
# col_to_compare_Deaths = 'countyFIPS'
# dfPop.rename(columns={col_to_compare_Pop: 'column_name'}, inplace=True)
# dfDeaths.rename(columns={col_to_compare_Deaths: 'column_name'}, inplace=True)

# # Merge the two datasets on the selected column
# merged_df = pd.merge(dfPop, dfDeaths, left_on='column_name', right_on='column_name', how='inner')

# # Count how many rows match
# num_matches = merged_df.shape[0]

# # Print the result
# print(f"There are {num_matches} rows that match on the '{col_to_compare_Pop}' column in dataset 1 and '{col_to_compare_Deaths}' column in dataset 2.")


#Merge the dataset on countyFIPS

# Merge the two datasets based on the County Code column
merged_df = pd.merge(dfPop, dfDeaths, on='countyFIPS', how='inner')

# Create a new column for the percentage of deaths in 2020
merged_df['deaths_pct_2020'] = merged_df['2020_total'] / merged_df['population'] * 100

# Remove rows with inf or NaN values
merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
merged_df = merged_df.dropna()

# Keep only the 'countyFIPS' and 'deaths_pct_2020' columns
merged_df = merged_df[['countyFIPS', 'deaths_pct_2020']]

# Save the updated dataset to a new file
merged_df.to_csv('death_percentage.csv', index=False)


#remove outliers
#switch labels
#only counties with higher population
