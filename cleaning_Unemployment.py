import pandas as pd

# Cleaning Unemployment.csv

# Cleaning Unemployment.csv

# Read in the CSV file
dfUnemployment = pd.read_csv(
    "/Users/margaretrivas/Desktop/IDC4140/finalProject/datasets/Unemployment.csv"
)
dfDeaths = pd.read_csv(
    "/Users/margaretrivas/Desktop/IDC4140/finalProject/datasets/deaths_by_year.csv"
)

# Select columns to keep
initial_keeping = ["FIPS_code", "Area_name", "State"]

# Filter the columns based on their names
cols_to_keep = initial_keeping + [
    col for col in dfUnemployment.columns if col.endswith("2020")
]
cleaned_df = dfUnemployment[cols_to_keep]

# Join the dataframes on the FIPS code columns and keep only the matching rows
merged_df = pd.merge(
    dfDeaths, cleaned_df, left_on="countyFIPS", right_on="FIPS_code", how="inner"
)

# Write the merged data to a new CSV file
merged_df.to_csv("clean_Unemployment.csv", index=False)
