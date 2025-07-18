# ================================
# Step 1: Import Libraries
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import joblib
from flask import Flask, request, jsonify

# ================================
# Step 2: Load and Clean Data
# ================================
df = pd.read_csv("unicorns till sep 2022.csv")

# Rename columns
df.rename(columns={
    'Company': 'Company',
    'Valuation ($B)': 'Valuation',
    'Date Joined': 'Date_Joined',
    'Country': 'Country',
    'City ': 'City',
    'Industry': 'Sector',
    'Investors': 'Investors'
}, inplace=True)

# Clean valuation column
df['Valuation'] = df['Valuation'].str.replace('$', '', regex=False).astype(float)

# Convert dates
df['Date_Joined'] = pd.to_datetime(df['Date_Joined'])
df['Month'] = df['Date_Joined'].dt.month
df['Year'] = df['Date_Joined'].dt.year

# Split investors
investors_split = df['Investors'].str.split(',', expand=True)
investors_split.columns = ['Investor_1', 'Investor_2', 'Investor_3', 'Investor_4']
df = pd.concat([df, investors_split], axis=1)

# Drop original Investors column
df.drop(columns='Investors', inplace=True)

# Handle missing values
df.fillna('None', inplace=True)

# ================================
# Step 3: Exploratory Data Analysis
# ================================
# Distribution of valuation
sns.histplot(df['Valuation'], kde=True)
plt.title("Distribution of Unicorn Valuation ($B)")
plt.show()

# Top sectors
top_sectors = df['Sector'].value_counts().head(10)
top_sectors.plot(kind='bar', title="Top 10 Sectors")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Country-wise valuation
country_val = df.groupby("Country")["Valuation"].sum().sort_values(ascending=False).head(10)
country_val.plot(kind='bar', title="Top 10 Countries by Unicorn Valuation")
plt.ylabel("Total Valuation ($B)")
plt.xticks(rotation=45)
plt.show()

# ================================
# Step 4: Feature Engineering
# ================================
df_encoded = pd.get_dummies(df[['Country', 'City', 'Sector', 'Year', 'Month']], drop_first=True)

X = df_encoded
y = df['Valuation']

# ================================
# Step 5: Train-Test Split & Scaling
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# Step 6: Model Training
# ================================
# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# Evaluation
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Random Forest R²:", r2_score(y_test, y_pred_rf))

# ================================
# Step 7: Hyperparameter Tuning
# ================================
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best Model
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_scaled)

print("Tuned Random Forest R²:", r2_score(y_test, y_pred_best))

# After training model
import joblib

joblib.dump(best_rf, "unicorn_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")  # ✅ Save column names

# ================================
# Step 8: Save Model
# ================================
joblib.dump(best_rf, "unicorn_model.pkl")

# ================================
# Step 9: Deployment with Flask (run separately)
# ================================
"""
app = Flask(__name__)
model = joblib.load("unicorn_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    return jsonify({'predicted_valuation': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
"""
