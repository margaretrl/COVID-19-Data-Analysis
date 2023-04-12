#Linear regression masks showing the correlation between wearing masks and deaths by county in year 2020
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dfMask = pd.read_csv('/Users/marijatravoric/IDC 4140/COVID-19-Data-Analysis/datasets/mask-use-by-county.csv')
dfDeath = pd.read_csv('/Users/marijatravoric/IDC 4140/COVID-19-Data-Analysis/datasets/deaths_by_year.csv')

#merge
# Rename column in dfMask to match column in dfDeath
dfMask = dfMask.rename(columns={'COUNTYFP': 'countyFIPS'})

# Merge the two datasets based on the County Code column
merged_df = pd.merge(dfMask, dfDeath, on='countyFIPS', how='inner')

# Select only the columns that you need
merged_df = merged_df[['countyFIPS', 'NEVER', 'RARELY', 'SOMETIMES', 'FREQUENTLY', 'ALWAYS', '2020_total']]

#checking if the number of rows match the number of matching rows from cleanMask (3142)
print(merged_df)

# Split the dataset into features (X) and target variable (y)
X = merged_df[['NEVER', 'RARELY', 'SOMETIMES', 'FREQUENTLY', 'ALWAYS']]
y = merged_df['2020_total']

# Create a Linear Regression object
reg = LinearRegression()

# Fit the model using the training data
reg.fit(X, y)

#Prints plots for all columns
fig, axs = plt.subplots(1, 5, figsize=(20,5))

for i, col in enumerate(['NEVER', 'RARELY', 'SOMETIMES', 'FREQUENTLY', 'ALWAYS']):
    X = merged_df[[col]]
    y = merged_df['2020_total']
    reg.fit(X, y)
    axs[i].scatter(X, y)
    axs[i].set_xlabel(f'Percentage of People \nWho {col.lower()} Wear Masks')
    axs[i].set_ylabel('COVID-19 Deaths in 2020')
    axs[i].set_title(f'Mask Usage vs. COVID-19 Deaths\n ({col})\nCoefficients: {reg.coef_[0]:.2f}')
    axs[i].set_ylim(0, max(y) + 5000)

plt.subplots_adjust(wspace=0.4)
plt.tight_layout()
plt.show()




#The coefficients represent the slope of the linear regression line, which gives us an idea of 
# #how strongly each feature (i.e., mask usage rate) is correlated with the target variable 
# #(i.e., number of deaths). A positive coefficient indicates a positive correlation 
# #(i.e., as mask usage rate increases, so does the number of deaths), while a negative coefficient 
# indicates a negative correlation (i.e., as mask usage rate increases, the number of deaths 
# decreases).

#why is increase in mask usage show increase in deaths.




########################################################################################################################################################################







