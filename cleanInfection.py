import pandas as pd
import datetime

## DATA CLEANING

# DEATHS

deaths_df = pd.read_csv(
    "/Users/margaretrivas/Desktop/IDC4140/finalProject/datasets/covid_deaths_usafacts.csv"
)

# create a new column with the sum of all values for 2020
deaths_df["2020_total"] = deaths_df.filter(regex="^2020", axis=1).sum(axis=1)

# create a new column with the sum of all values for 2020
deaths_df["2021_total"] = deaths_df.filter(regex="^2021", axis=1).sum(axis=1)

deaths_df["2022_total"] = deaths_df.filter(regex="^2022", axis=1).sum(axis=1)

# save the updated DataFrame to a new CSV file
deaths_df.to_csv("deaths_by_year.csv", index=False)

deaths_df = pd.read_csv(
    "/Users/margaretrivas/Desktop/IDC4140/finalProject/deaths_by_year.csv"
)

df = deaths_df[["countyFIPS", "State", "2020_total", "2021_total", "2022_total"]]

# Save the updated data to a new CSV file
df.to_csv("deaths_by_year_updated.csv", index=False)


## CASES

cases_df = pd.read_csv(
    "/Users/margaretrivas/Desktop/IDC4140/finalProject/datasets/covid_confirmed_usafacts.csv"
)

# create a new column with the sum of all values for 2020
cases_df["2020_total"] = cases_df.filter(regex="^2020", axis=1).sum(axis=1)

# create a new column with the sum of all values for 2020
cases_df["2021_total"] = cases_df.filter(regex="^2021", axis=1).sum(axis=1)

cases_df["2022_total"] = cases_df.filter(regex="^2022", axis=1).sum(axis=1)

# save the updated DataFrame to a new CSV file
cases_df.to_csv("cases_by_year.csv", index=False)

cases_df = pd.read_csv(
    "/Users/margaretrivas/Desktop/IDC4140/finalProject/cases_by_year.csv"
)

df = cases_df[["countyFIPS", "State", "2020_total", "2021_total", "2022_total"]]

# Save the updated data to a new CSV file
df.to_csv("confirmed_cases_by_year_updated.csv", index=False)
