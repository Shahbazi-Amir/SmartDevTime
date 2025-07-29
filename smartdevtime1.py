# Step 1: Import required libraries
import pandas as pd

# Step 2: Load the dataset
df = pd.read_csv("dataset/fp.csv")

# Step 3: Show the first few rows
df.head()



# Step 1: نمایش اطلاعات کلی دیتافریم
df.info()

# Step 2: بررسی تعداد داده‌های گمشده در هر ستون
df.isnull().sum()

# Step 3: خلاصه آماری از ستون‌های عددی
df.describe()

'''<class 'pandas.core.frame.DataFrame'>
RangeIndex: 81 entries, 0 to 80
Data columns (total 13 columns):
 #   Column           Non-Null Count  Dtype
---  ------           --------------  -----
 0   id               81 non-null     int64
 1   Project          81 non-null     int64
 2   TeamExp          81 non-null     int64
 3   ManagerExp       81 non-null     int64
 4   YearEnd          81 non-null     int64
 5   Length           81 non-null     int64
 6   Effort           81 non-null     int64
 7   Transactions     81 non-null     int64
 8   Entities         81 non-null     int64
 9   PointsNonAdjust  81 non-null     int64
 10  Adjustment       81 non-null     int64
 11  PointsAjust      81 non-null     int64
 12  Language         81 non-null     int64
dtypes: int64(13)
memory usage: 8.4 KB'''

import matplotlib.pyplot as plt
import seaborn as sns

# Plot distribution of each numerical feature
numeric_cols = ['TeamExp', 'ManagerExp', 'Effort', 'Transactions', 'Entities', 
                'PointsNonAdjust', 'Adjustment', 'PointsAjust']

plt.figure(figsize=(16, 12))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()



# Check how many -1 and 0 values exist in each column
for col in df.columns:
    if df[col].dtype == 'int64':
        n_minus1 = (df[col] == -1).sum()
        n_zero = (df[col] == 0).sum()
        if n_minus1 > 0 or n_zero > 0:
            print(f"{col}: -1 count = {n_minus1}, 0 count = {n_zero}")



'''TeamExp: -1 count = 2, 0 count = 7
ManagerExp: -1 count = 3, 0 count = 5'''



# Remove rows with -1 in TeamExp or ManagerExp
df_cleaned = df[(df['TeamExp'] != -1) & (df['ManagerExp'] != -1)]
print(f"Remaining rows after cleaning: {df_cleaned.shape[0]}")



'''Remaining rows after cleaning: 77'''


# 📊 Correlation Matrix and Heatmap
# Compute correlation matrix
corr_matrix = df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Feature Correlation Matrix')
plt.show()



# 🔍 Correlation of all features with Effort (sorted)

correlations = df.corr()['Effort'].sort_values(ascending=False)
print(correlations)


'''Effort             1.000000
PointsAjust        0.738271
PointsNonAdjust    0.705449
Length             0.693280
Transactions       0.581881
Entities           0.510328
Adjustment         0.463865
ManagerExp         0.158303
id                 0.126153
Project            0.126153
TeamExp            0.119529
YearEnd           -0.048367
Language          -0.261942
Name: Effort, dtype: float64'''



# Step 1: Remove unhelpful columns
df_model = df_cleaned.drop(columns=['id', 'Project', 'YearEnd'])

# Step 2: Show the shape and first few rows of the cleaned dataframe
print("Shape after column removal:", df_model.shape)
df_model.head()




# Step 1: Separate features (X) and target (y)
X = df_model.drop(columns=['Effort'])  # Features: all columns except Effort
y = df_model['Effort']                # Target: the column we want to predict

# Step 2: Split into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Print the shape of each set
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


'''Training set shape: (61, 9)
Test set shape: (16, 9)'''




# Step 1: Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Step 2: Standardize features (fit only on training data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Step 4: Predict on test data
y_pred = lr_model.predict(X_test_scaled)

# Step 5: Evaluate model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²:", round(r2, 2))



'''MAE: 2103.16
RMSE: 2882.91
R²: 0.42'''



# Step 1: Plot actual vs predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='steelblue', s=70)

# Step 2: Plot ideal line (y = x)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')

# Step 3: Labels and title
plt.xlabel("Actual Effort")
plt.ylabel("Predicted Effort")
plt.title("Actual vs Predicted Effort (Linear Regression)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Create and train the model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Predict on test set
y_pred_tree = tree_model.predict(X_test)




# Evaluate the model
mae_tree = mean_absolute_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
r2_tree = r2_score(y_test, y_pred_tree)

print("Decision Tree Results:")
print(f"MAE: {mae_tree:.2f}")
print(f"RMSE: {rmse_tree:.2f}")
print(f"R²: {r2_tree:.2f}")



'''Decision Tree Results:
MAE: 2779.00
RMSE: 3601.57
R²: 0.09'''


plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_tree, color='darkorange', s=70)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel("Actual Effort")
plt.ylabel("Predicted Effort")
plt.title("Actual vs Predicted Effort (Decision Tree)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Get feature names if X is a DataFrame
feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'X{i}' for i in range(X_train.shape[1])]

# Get coefficients from the trained linear regression model
coefficients = lr_model.coef_


# Combine into a DataFrame for easier interpretation
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(coef_df)



'''          Feature   Coefficient
7      PointsAjust  10120.003320
3     Transactions  -3329.217373
5  PointsNonAdjust  -3273.369574
4         Entities  -1527.084661
2           Length   1283.416587
8         Language  -1242.888855
6       Adjustment   -979.422520
0          TeamExp   -365.885652
1       ManagerExp    232.266398'''


important_features = ['PointsAjust', 'Transactions', 'PointsNonAdjust', 'Entities', 'Length', 'Language']


X_train_simple = X_train[important_features]
X_test_simple = X_test[important_features]


from sklearn.linear_model import LinearRegression

simple_model = LinearRegression()
simple_model.fit(X_train_simple, y_train)


simple_coefficients = simple_model.coef_
simple_coef_df = pd.DataFrame({
    'Feature': important_features,
    'Coefficient': simple_coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(simple_coef_df)



''' Feature  Coefficient
5         Language -1481.000536
4           Length   166.254540
0      PointsAjust    19.723203
2  PointsNonAdjust    -5.117410
1     Transactions    -2.923670
3         Entities    -2.193740'''


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred_simple = simple_model.predict(X_test_simple)

mae = mean_absolute_error(y_test, y_pred_simple)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_simple))
r2 = r2_score(y_test, y_pred_simple)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")



'''MAE: 2024.28
RMSE: 2831.22
R²: 0.44'''




from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Create the XGBoost regressor (with random_state for reproducibility)
xgb_model = XGBRegressor(random_state=42)

# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

# Run grid search on training data
grid_search.fit(X_train, y_train)



# Step 1: Import XGBoost Regressor and evaluation metrics
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Initialize the XGBoost regressor with basic parameters
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)

# Step 3: Fit the model on training data (scaled or original)
xgb_model.fit(X_train, y_train)

# Step 4: Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Step 5: Evaluate the model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost MAE: {mae_xgb:.2f}")
print(f"XGBoost RMSE: {rmse_xgb:.2f}")
print(f"XGBoost R²: {r2_xgb:.2f}")



from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Create the XGBoost regressor (with random_state for reproducibility)
xgb_model = XGBRegressor(random_state=42)

# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

# Run grid search on training data
grid_search.fit(X_train, y_train)





plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_xgb, color='green', s=70)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel("Actual Effort")
plt.ylabel("Predicted Effort")
plt.title("Actual vs Predicted Effort (XGBoost Regression)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Create the XGBoost regressor (with random_state for reproducibility)
xgb_model = XGBRegressor(random_state=42)

# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

# Run grid search on training data
grid_search.fit(X_train, y_train)



plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_xgb, color='green', s=70)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
plt.xlabel("Actual Effort")
plt.ylabel("Predicted Effort")
plt.title("Actual vs Predicted Effort (XGBoost Regression)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# Step 1: Show best parameters
print("Best Parameters Found:")
print(grid_search.best_params_)

# Step 2: Show best R² score from cross-validation
print("\nBest R² Score from CV:")
print(round(grid_search.best_score_, 4))

# Step 3: Extract the best model
best_xgb_model = grid_search.best_estimator_




# Step 1: Predict on test set using the best model
y_pred_xgb = best_xgb_model.predict(X_test)

# Step 2: Evaluate performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"MAE: {mae_xgb:.2f}")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"R²: {r2_xgb:.2f}")




# Step 1: Remove rows with -1 in TeamExp or ManagerExp (as done earlier)
df_cleaned = df[(df['TeamExp'] != -1) & (df['ManagerExp'] != -1)]

# Step 2: Drop irrelevant columns
df_model = df_cleaned.drop(columns=['id', 'Project', 'YearEnd'])

# Step 3: Apply one-hot encoding to the 'Language' column
df_encoded = pd.get_dummies(df_model, columns=['Language'], drop_first=True)

# Optional: Show first few rows
df_encoded.head()




# Step 1: Create interaction term
df_encoded['Team_Manager_Interaction'] = df_encoded['TeamExp'] * df_encoded['ManagerExp']

# Step 2: Show first few rows to verify
df_encoded[['TeamExp', 'ManagerExp', 'Team_Manager_Interaction']].head()






df_encoded['Team_Manager_Interaction'] = df_encoded['TeamExp'] * df_encoded['ManagerExp']



import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Set figure
plt.figure(figsize=(8, 6))

# Step 2: Plot scatter with color and size based on Effort
sns.scatterplot(
    data=df_encoded,
    x='TeamExp',
    y='ManagerExp',
    size='Effort',
    hue='Effort',
    palette='viridis',
    legend='brief'
)

# Step 3: Customize
plt.title('TeamExp vs ManagerExp colored by Effort')
plt.xlabel('Team Experience')
plt.ylabel('Manager Experience')
plt.grid(True)
plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Prepare figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Step 2: Plot 3D scatter
ax.scatter(
    df_encoded['TeamExp'],
    df_encoded['ManagerExp'],
    df_encoded['Effort'],
    c=df_encoded['Effort'],
    cmap='viridis',
    s=50
)

# Step 3: Labels and title
ax.set_xlabel('Team Experience')
ax.set_ylabel('Manager Experience')
ax.set_zlabel('Effort')
ax.set_title('3D View: Effort vs TeamExp × ManagerExp')

plt.tight_layout()
plt.show()




from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Separate features and target
X = df_encoded.drop(columns=['Effort'])
y = df_encoded['Effort']

# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = lr_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")




from sklearn.linear_model import Ridge

# Step 1: Define and train the Ridge model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Step 2: Predict on test set
y_pred_ridge = ridge_model.predict(X_test)

# Step 3: Evaluate
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge MAE: {mae_ridge:.2f}")
print(f"Ridge RMSE: {rmse_ridge:.2f}")
print(f"Ridge R²: {r2_ridge:.2f}")



from sklearn.linear_model import Lasso

# Step 1: Define and train the Lasso model
lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

# Step 2: Predict on test set
y_pred_lasso = lasso_model.predict(X_test)

# Step 3: Evaluate
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Lasso MAE: {mae_lasso:.2f}")
print(f"Lasso RMSE: {rmse_lasso:.2f}")
print(f"Lasso R²: {r2_lasso:.2f}")



from sklearn.ensemble import RandomForestRegressor

# Step 1: Create and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 2: Predict on test set
y_pred_rf = rf_model.predict(X_test)

# Step 3: Evaluate
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MAE: {mae_rf:.2f}")
print(f"Random Forest RMSE: {rmse_rf:.2f}")
print(f"Random Forest R²: {r2_rf:.2f}")




