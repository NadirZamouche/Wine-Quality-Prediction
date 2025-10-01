# Libraries
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1 Data Collection
# Load datasets
df1 = pd.read_csv("../raw/winequality-red.csv", sep=";")
df1.head()
df2 = pd.read_csv("../raw/winequality-white.csv", sep=";")
df2.head()
# Insert wine_color column just before the last column (quality)
last_index = df1.shape[1] - 1  # index of last column
df1.insert(last_index, "wine color", "red")
df2.insert(last_index, "wine color", "white")
# checking out the new column
df1.head()
df2.head()
# Merge
df = pd.concat([df1, df2], ignore_index=True)
df.head()

# 2 Data Inspection
df.info()
df.describe()
df.value_counts("quality")

# 3 Data Cleaning - no dups found
duplicates = df.duplicated()
# Count duplicates
print("Number of duplicate rows:", duplicates.sum())
# Show actual duplicate rows
print(df[df.duplicated()])

# 4 String Data Transformation for "wine color" column using label encoder
df["wine color"] = LabelEncoder().fit_transform(df["wine color"])
df.head()
df.tail()

# 5 Exploartory Data Analysis EDA & Visualization
# 5.1 Box Plot
# Columns to check for outliers (12 "quality" is not included)
columns_selection = df.columns[0:12]


# Define a function to identify outliers using IQR method
def identify_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))


# Create subplots for each column
fig, axes = plt.subplots(
    nrows=len(columns_selection), ncols=1, figsize=(8, 4 * len(columns_selection))
)
fig.subplots_adjust(hspace=0.5)
# Loop through each selected column
for i, column in enumerate(columns_selection):
    # Draw box plot
    axes[i].boxplot(df[column])
    axes[i].set_title(f"Box Plot for {column}")
    axes[i].set_ylabel(column)
# Show the box plots and outliers
plt.show()

# 5.2 Histograms
# Calculate the number of rows and columns for subplots
num_columns = 3
num_rows = math.ceil(len(columns_selection) / num_columns)
# Create subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(15, 4 * num_rows))
fig.subplots_adjust(hspace=0.5)
# Loop through each selected column
for i, column in enumerate(columns_selection):
    row_num = i // num_columns
    col_num = i % num_columns
    # Plot histogram with bins=50
    df[column].hist(bins=50, ax=axes[row_num, col_num])
    axes[row_num, col_num].set_title(f"Histogram for {column}")
    axes[row_num, col_num].set_xlabel(column)
    axes[row_num, col_num].set_ylabel("Frequency")
# Remove any empty subplots
for i in range(len(columns_selection), num_rows * num_columns):
    fig.delaxes(axes.flatten()[i])
# Show the histograms
plt.show()

# 5.3 Correlation Matrix
# calculate correlation matrix
corrmat = df.corr()
# select column names for plotting
top_corr_features = corrmat.index
# plot heat map
plt.figure(figsize=(13, 13))
g = sns.heatmap(
    corrmat[top_corr_features].loc[top_corr_features], annot=True, cmap="RdBu"
)
plt.show()

# 6 Feature Scaling
# Extract the other column names (excluding the last two)
columns_to_scale = df.columns[:-2]
# Initialize the StandardScaler
scaler = StandardScaler()
# Standard scale the selected columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
# Display the DataFrame
df.head()

# 7 Save DataFrame as CSV
df.to_csv("../processed/winequality_merged.csv", index=False)
