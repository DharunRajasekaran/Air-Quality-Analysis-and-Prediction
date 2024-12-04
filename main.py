import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Dataset created directly in code
data = {
    "City": ["New York", "Los Angeles", "Beijing", "Mumbai", "Sydney", "London", "Tokyo", "Cairo"] * 10,
    "Date": pd.date_range(start="2023-01-01", periods=80).strftime("%Y-%m-%d").tolist(),
    "PM2.5": np.random.uniform(10, 150, 80),
    "PM10": np.random.uniform(20, 200, 80),
    "NO2": np.random.uniform(5, 50, 80),
    "SO2": np.random.uniform(2, 30, 80),
    "CO": np.random.uniform(0.1, 2.0, 80),
    "O3": np.random.uniform(10, 60, 80),
    "Temperature": np.random.uniform(-10, 40, 80),
    "Humidity": np.random.uniform(20, 90, 80),
    "AQI": np.random.uniform(50, 300, 80).astype(int),
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print("Dataset Sample:\n", df.head())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Convert Date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Extract Year, Month, and Day
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Exploratory Data Analysis
print("\nDataset Statistics:\n", df.describe())

# Visualize pollutant trends across cities
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Date", y="PM2.5", hue="City")
plt.title("PM2.5 Levels Over Time")
plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Visualize AQI Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df["AQI"], kde=True, bins=20)
plt.title("AQI Distribution")
plt.xlabel("Air Quality Index (AQI)")
plt.ylabel("Frequency")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
#numerical_df = df.select_dtypes(include=['number'])
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Air Quality Data")
plt.show()

# Clustering cities based on average pollutant levels
avg_pollutants = df.groupby("City")[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]].mean()
print("\nAverage Pollutant Levels by City:\n", avg_pollutants)

# Visualization: Average pollutant levels by city
avg_pollutants.plot(kind="bar", figsize=(12, 6))
plt.title("Average Pollutant Levels by City")
plt.xlabel("City")
plt.ylabel("Pollutant Concentration")
plt.legend(title="Pollutants")
plt.tight_layout()
plt.show()

# Feature Engineering for AQI Prediction
X = df[["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity"]]
y = df["AQI"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict AQI
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:\nMean Absolute Error (MAE): {mae:.2f}\nR-squared (R2): {r2:.2f}")

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_,
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
plt.title("Feature Importance for AQI Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Time-Series Analysis: Monthly Average AQI
monthly_avg_aqi = df.groupby(["Year", "Month"])["AQI"].mean().reset_index()
monthly_avg_aqi["Date"] = pd.to_datetime(monthly_avg_aqi[["Year", "Month"]].assign(Day=1))

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg_aqi, x="Date", y="AQI", marker="o")
plt.title("Monthly Average AQI")
plt.xlabel("Date")
plt.ylabel("Average AQI")
plt.show()

# Predict Future AQI Levels (Time-Series Forecasting Placeholder)
future_dates = pd.date_range(start="2024-01-01", periods=10, freq="M")
future_data = pd.DataFrame({
    "Date": future_dates,
    "PM2.5": np.random.uniform(10, 150, 10),
    "PM10": np.random.uniform(20, 200, 10),
    "NO2": np.random.uniform(5, 50, 10),
    "SO2": np.random.uniform(2, 30, 10),
    "CO": np.random.uniform(0.1, 2.0, 10),
    "O3": np.random.uniform(10, 60, 10),
    "Temperature": np.random.uniform(-10, 40, 10),
    "Humidity": np.random.uniform(20, 90, 10),
})

future_data["AQI_Predicted"] = model.predict(future_data.drop(columns=["Date"]))

print("\nFuture AQI Predictions:\n", future_data)

# Visualization: Future AQI Predictions
plt.figure(figsize=(10, 6))
sns.lineplot(data=future_data, x="Date", y="AQI_Predicted", marker="o", color="red")
plt.title("Predicted Future AQI Levels")
plt.xlabel("Date")
plt.ylabel("Predicted AQI")
plt.show()
