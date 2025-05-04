#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import scipy as sp


# In[14]:


df = pd.read_csv(r"C:\Users\mkand\Downloads\archive\used_cars.csv")

# removing index column
df = df.iloc[: , 1:]

# Checking the first 5 entries of dataset
df.head()



# In[15]:


headers = [
    "model", "model_year", "mileage", "fuel_type",
    "engine", "transmission", "ext_col", "int_col",
    "accident", "clean_title", "price"
]

df.columns=headers
df.head()


# In[16]:


data = df

# Finding the missing values
data.isna().any()

# Finding if missing values
data.isnull().any()


# In[19]:


def clean_data(df):
    # Convert mileage to string first, then clean and convert to integer
    df["mileage"] = df["mileage"].astype(str).str.replace(" mi.", "").str.replace(",", "").astype(int)

    # Extract only the engine size (first number followed by "L")
    df["engine"] = df["engine"].astype(str).str.extract(r"(\d+\.\d+L)")

    # Standardize transmission type
    df["transmission"] = df["transmission"].astype(str).str.extract(r"(\bAutomatic\b|\bManual\b)")

    # Convert accident reports to "Yes" or "No"
    df["accident"] = df["accident"].astype(str).apply(lambda x: "No" if x == "None reported" else "Yes")

    # Drop 'clean_title' if it exists in the DataFrame
    if "clean_title" in df.columns:
        df.drop(columns=["clean_title"], inplace=True)

    # Convert price to float (remove "$" and commas)
    df["price"] = df["price"].astype(str).str.replace("$", "").str.replace(",", "").astype(float)

    return df

# Apply cleaning
df_cleaned = clean_data(df)

# Display cleaned dataset info and first few rows
df_cleaned.info(), df_cleaned.head()



# In[20]:


# Apply Simple Feature Scaling
df_cleaned["mileage_scaled"] = df_cleaned["mileage"] / df_cleaned["mileage"].max()
df_cleaned["price_scaled"] = df_cleaned["price"] / df_cleaned["price"].max()

# Define binning categories
mileage_bins = [0, 30000, 70000, df_cleaned["mileage"].max()]
mileage_labels = ["Low", "Medium", "High"]
df_cleaned["mileage_category"] = pd.cut(df_cleaned["mileage"], bins=mileage_bins, labels=mileage_labels, include_lowest=True)

price_bins = [0, 20000, 50000, df_cleaned["price"].max()]
price_labels = ["Budget", "Mid-Range", "Luxury"]
df_cleaned["price_category"] = pd.cut(df_cleaned["price"], bins=price_bins, labels=price_labels, include_lowest=True)

# Display the updated dataframe
df_cleaned[["mileage", "mileage_scaled", "mileage_category", "price", "price_scaled", "price_category"]].head()


# In[21]:


# Identify categorical columns
categorical_cols = ["model", "fuel_type", "engine", "transmission", "ext_col", "int_col", "accident"]
existing_cols = [col for col in categorical_cols if col in df_cleaned.columns]  # Ensure column exists

label_encoders = {}
for col in existing_cols:
    le = LabelEncoder()
    df_cleaned[col + "_num"] = le.fit_transform(df_cleaned[col].astype(str))
    label_encoders[col] = le

# Check the updated dataframe
df_cleaned.head()


# In[22]:


# Set style
sns.set_theme(style="whitegrid")

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Boxplot: Price vs. Fuel Type
sns.boxplot(x="fuel_type", y="price", data=df_cleaned, ax=axes[0])
axes[0].set_title("Price Distribution by Fuel Type")
axes[0].set_xlabel("Fuel Type")
axes[0].set_ylabel("Price ($)")

# Bar chart: Average Price by Mileage Category
sns.barplot(x="mileage_category", y="price", data=df_cleaned, ax=axes[1], estimator=lambda x: sum(x) / len(x))
axes[1].set_title("Average Price by Mileage Category")
axes[1].set_xlabel("Mileage Category")
axes[1].set_ylabel("Average Price ($)")

# Bar chart: Average Price by Transmission Type
sns.barplot(x="transmission", y="price", data=df_cleaned, ax=axes[2], estimator=lambda x: sum(x) / len(x))
axes[2].set_title("Average Price by Transmission Type")
axes[2].set_xlabel("Transmission")
axes[2].set_ylabel("Average Price ($)")

# Show plots
plt.tight_layout()
plt.show()


# In[23]:


grouped_data = df.groupby(["fuel_type", "transmission", "mileage_category"], observed=False).agg(
    avg_price=("price", "mean"),
    count=("price", "count")
).reset_index()

# Sort by price and reset index
grouped_data = grouped_data.sort_values(by="avg_price", ascending=False).reset_index(drop=True)

# Display grouped data
grouped_data.head()  # Ensure correct indexing



# In[24]:


# Pivot the data: fuel_type as rows, transmission as columns, avg_price as values
pivot_table = grouped_data.pivot_table(index="fuel_type", columns="transmission", values="avg_price")

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".0f", linewidths=0.5)

# Titles and labels
plt.title("Average Price Heatmap by Fuel Type & Transmission")
plt.xlabel("Transmission Type")
plt.ylabel("Fuel Type")

# Show the plot
plt.show()


# In[25]:


# Plot regression line for price vs. mileage (or another numerical feature)
plt.figure(figsize=(8, 6))
sns.regplot(x=df["mileage"], y=df["price"], scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})

# Titles and labels
plt.title("Price vs. Mileage - Linear Relationship")
plt.xlabel("Mileage")
plt.ylabel("Price ($)")

# Show the plot
plt.show()


# In[34]:


# Load cleaned dataset
df = pd.read_csv(r"C:\Users\mkand\Downloads\cleaned_used_cars.csv")

# Display first few rows
print(df.head())



# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load data
df = pd.read_csv(r"C:\Users\mkand\Downloads\cleaned_used_cars.csv")

# Feature Engineering
df['car_age'] = 2025 - df['model_year']
df['mileage_per_year'] = df['mileage'] / df['car_age'].replace(0, 1)
df['log_price'] = np.log1p(df['price'])
df['mileage_to_age_ratio'] = df['mileage'] / (df['car_age'] + 1)

# Polynomial Features
df['mileage_squared'] = df['mileage'] ** 2
df['mileage_per_year_squared'] = df['mileage_per_year'] ** 2

# Remove outliers using RobustScaler
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df[['price', 'mileage_per_year', 'mileage_to_age_ratio']])
price_mileage_scaled = pd.DataFrame(df_scaled, columns=['price_scaled', 'mileage_per_year_scaled', 'mileage_to_age_ratio_scaled'])

Q1 = price_mileage_scaled.quantile(0.25)
Q3 = price_mileage_scaled.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
mask = ((price_mileage_scaled >= lower_bound) & (price_mileage_scaled <= upper_bound)).all(axis=1)
df = df[mask]

# Drop rows with missing target values
df = df.dropna(subset=['price'])

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# Preprocessing pipelines
numerical_transformer = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Define model with stronger regularization
model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1)

# Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform randomized search with increased parameter space
param_distributions = {
    'model__n_estimators': np.arange(300, 501, 50),
    'model__learning_rate': np.linspace(0.005, 0.15, 20),
    'model__max_depth': [7, 10, 12, 15],
    'model__min_child_weight': [1, 3, 5, 7],
    'model__subsample': [0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 1.0],
    'model__reg_alpha': [5.0, 10.0, 15.0],
    'model__reg_lambda': [10.0, 15.0, 20.0],
    'model__gamma': [0, 0.25, 1.0],
    'model__scale_pos_weight': [1, 2, 3]
}

random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=50, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, error_score='raise', random_state=42)
random_search.fit(X_train, y_train)

# Best parameters and model evaluation
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Best Parameters: {random_search.best_params_}')
print(f'Mean Absolute Error: {mae}')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Load cleaned dataset
df = pd.read_csv("cleaned_used_cars.csv")

# Add brand extraction (e.g., first word from model name)
df['brand'] = df['brand'].str.split().str[0]

# Feature Engineering
df['car_age'] = 2025 - df['model_year']
df['mileage_per_year'] = df['mileage'] / df['car_age'].replace(0, 1)
df['mileage_to_age_ratio'] = df['mileage'] / (df['car_age'] + 1)
df['mileage_squared'] = df['mileage'] ** 2
df['mileage_per_year_squared'] = df['mileage_per_year'] ** 2

# Remove outliers using RobustScaler
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df[['price', 'mileage_per_year', 'mileage_to_age_ratio']])
price_mileage_scaled = pd.DataFrame(df_scaled, columns=['price_scaled', 'mileage_per_year_scaled', 'mileage_to_age_ratio_scaled'])

Q1 = price_mileage_scaled.quantile(0.25)
Q3 = price_mileage_scaled.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
mask = ((price_mileage_scaled >= lower_bound) & (price_mileage_scaled <= upper_bound)).all(axis=1)
df = df[mask]

# Drop rows with missing target values
df = df.dropna(subset=['price'])

# Define features and target
X = df.drop(['price', 'log_price'], axis=1, errors='ignore')  # Drop log_price if exists
y = df['price']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns

# Preprocessing pipelines
numerical_transformer = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Model
model = XGBRegressor(random_state=42, tree_method='hist', n_jobs=-1)

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_distributions = {
    'model__n_estimators': np.arange(300, 501, 50),
    'model__learning_rate': np.linspace(0.005, 0.15, 20),
    'model__max_depth': [7, 10, 12, 15],
    'model__min_child_weight': [1, 3, 5, 7],
    'model__subsample': [0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.7, 0.8, 1.0],
    'model__reg_alpha': [5.0, 10.0, 15.0],
    'model__reg_lambda': [10.0, 15.0, 20.0],
    'model__gamma': [0, 0.25, 1.0],
    'model__scale_pos_weight': [1, 2, 3]
}

random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=50, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, error_score='raise', random_state=42)
random_search.fit(X_train, y_train)

# Results
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Best Parameters: {random_search.best_params_}')
print(f'Mean Absolute Error: {mae:.2f}')




