import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Models for unenployment rates vs covid deaths & cases for 2020

# Load
# Deaths totals are also inside this new cleaned csv
dfUnenmployment = pd.read_csv(
    "/Users/margaretrivas/Desktop/IDC4140/finalProject/clean_Unemployment.csv"
)

dfCases = pd.read_csv(
    "/Users/margaretrivas/Desktop/IDC4140/finalProject/datasets/confirmed_cases_by_year_updated.csv"
)


##########################################
### LINEAR REGRESSION MODEL FOR DEATHS ###
##########################################

# Split the dataset into features (X) and target variable (y)
X = dfUnenmployment["2020_total"].values.reshape(-1, 1)
y = dfUnenmployment["Unemployment_rate_2020"].values.reshape(-1, 1)

# Handle missing values in y using mean imputation
imputer = SimpleImputer(strategy="mean")
y = imputer.fit_transform(y)

# Create a linear regression model and fit it to the data
model = LinearRegression().fit(X, y)

# Print the model's coefficients
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Predict the unemployment rate for a given 2020_total value
new_X = np.array([[10000], [20000], [30000]])
predicted_y = model.predict(new_X)
print("Predicted unemployment rate for [10000, 20000, 30000]:", predicted_y.flatten())
# Plot the scatter plot of X and y
plt.scatter(X, y, alpha=0.5)

# Plot the line of best fit obtained from the linear regression model
x_min = np.min(X)
x_max = np.max(X)
x_range = np.linspace(x_min, x_max, 100)
y_range = model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_range, color="red")

# Set the axis labels and title
plt.xlabel("Total Deaths 2020")
plt.ylabel("Unemployment rate 2020")
plt.title("Scatter plot and line of best fit")

# Display the plot
plt.show()

#####################################
#### CLUSTERING MODEL FOR DEATHS ####
#####################################

# Extract the columns of interest
X = dfUnenmployment[["2020_total", "Unemployment_rate_2020"]].values

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a KMeans clustering model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the scaled data
kmeans.fit(X_scaled)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the data points colored by their cluster label
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.5)

# Plot the centroids as black crosses
plt.scatter(
    centroids[:, 0], centroids[:, 1], marker="x", s=200, linewidths=3, color="black"
)

# Set the axis labels and title
plt.xlabel("Total Deaths 2020")
plt.ylabel("Unemployment rate 2020")
plt.title("KMeans Clustering")

# Display the plot
plt.show()


##########################################
### LINEAR REGRESSION MODEL FOR CASES ###
##########################################

# Split the dataset into features (X) and target variable (y)
X = dfCases["2020_total"].values.reshape(-1, 1)
# Remove the last row from X1 to match the number of rows in X2
X = X[:-1]
y = dfUnenmployment["Unemployment_rate_2020"].values.reshape(-1, 1)

# Handle missing values in y using mean imputation
imputer = SimpleImputer(strategy="mean")
y = imputer.fit_transform(y)

# Create a linear regression model and fit it to the data
model = LinearRegression().fit(X, y)

# Print the model's coefficients
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Predict the unemployment rate for a given 2020_total value
new_X = np.array([[10000], [20000], [30000]])
predicted_y = model.predict(new_X)
print("Predicted unemployment rate for [10000, 20000, 30000]:", predicted_y.flatten())
# Plot the scatter plot of X and y
plt.scatter(X, y, alpha=0.5)

# Plot the line of best fit obtained from the linear regression model
x_min = np.min(X)
x_max = np.max(X)
x_range = np.linspace(x_min, x_max, 100)
y_range = model.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_range, color="red")

# Set the axis labels and title
plt.xlabel("Total Cases 2020")
plt.ylabel("Unemployment rate 2020")
plt.title("Scatter plot and line of best fit")

# Display the plot
plt.show()

#####################################
#### CLUSTERING MODEL FOR CASES ####
#####################################

## FIX SHAPE


# Extract the columns of interest from each dataframe
X1 = dfUnenmployment[["Unemployment_rate_2020"]].values
X2 = dfCases[["2020_total"]].values
# Remove the last row from X1 to match the number of rows in X2
X1 = X1[:-1]
# Concatenate the two arrays vertically
# X = np.concatenate((X1, X2), axis=0)
X = np.vstack((X1, X2))

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a KMeans clustering model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model to the scaled data
kmeans.fit(X_scaled)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the data points colored by their cluster label
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.5)

# Plot the centroids as black crosses
plt.scatter(
    centroids[:, 0], centroids[:, 1], marker="x", s=200, linewidths=3, color="black"
)

# Set the axis labels and title
plt.xlabel("Total Cases 2020")
plt.ylabel("Unemployment rate 2020")
plt.title("KMeans Clustering")

# Display the plot
plt.show()
