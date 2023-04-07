#Cleaning mask-use-by-county.csv
import pandas as pd

dfMask = pd.read_csv('/Users/marijatravoric/IDC 4140/COVID-19-Data-Analysis/datasets/mask-use-by-county.csv')

dfDeaths = pd.read_csv('/Users/marijatravoric/IDC 4140/COVID-19-Data-Analysis/datasets/covid_deaths_usafacts.csv')

#########################
# Keeping all 2020 info #
#########################

# Select columns to keep
columns_to_keep = ['countyFIPS', 'County Name', 'State', 'StateFIPS']

# Select columns to delete that don't start with 2020
columns_to_delete = dfDeaths.filter(regex='^(?!2020).').columns

# Exclude specific columns from the list of columns to delete
columns_to_delete = [col for col in columns_to_delete if col not in columns_to_keep]

# Drop the columns to delete from the dataset
dfDeaths.drop(columns_to_delete, axis=1, inplace=True)

#check for null values
num_null_values = dfDeaths.isnull().sum().sum()

if num_null_values == 0:
    print("The updated dataset does not contain any null values.")
else:
    print(f"The updated dataset contains {num_null_values} null values.")

# Save the updated dataset to a new file
dfDeaths.to_csv('updated_Deaths_dataset.csv', index=False)





#####################################################
# Checking how many rows match between each dataset #
#####################################################

#Select the columns to compare and rename them to the same name
col_to_compare_Mask = 'COUNTYFP'
col_to_compare_Deaths = 'countyFIPS'
dfMask.rename(columns={col_to_compare_Mask: 'column_name'}, inplace=True)
dfDeaths.rename(columns={col_to_compare_Deaths: 'column_name'}, inplace=True)

# Merge the two datasets on the selected column
merged_df = pd.merge(dfMask, dfDeaths, left_on='column_name', right_on='column_name', how='inner')

# Count how many rows match
num_matches = merged_df.shape[0]

# Print the result
print(f"There are {num_matches} rows that match on the '{col_to_compare_Mask}' column in dataset 1 and '{col_to_compare_Deaths}' column in dataset 2.")

