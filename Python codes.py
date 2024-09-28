import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'D:/Research/Data/Analysis/Final analysis.xlsx'  # Replace with the actual path to your Excel file
df = pd.read_excel(file_path)

# Ensure the dataset has the required columns for prediction
relevant_columns = ['Well No', 'Elevation', 'NDVI', 'LST', 'iso14', 'iso1430', 'iso3060', 'iso6090', 'GWL']
df_filtered = df[relevant_columns].dropna()

# Define features (X) and target (y)
X = df_filtered[['Elevation', 'NDVI', 'LST', 'iso14', 'iso1430', 'iso3060', 'iso6090']]
y = df_filtered['GWL']

# Split the dataset into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train the Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Performance:\nMSE: {mse:.4f}\nR-squared: {r2:.4f}")

# Function to predict GWL based on user input
def predict_gwl(elevation, ndvi, lst, iso14, iso1430, iso3060, iso6090):
    input_data = np.array([[elevation, ndvi, lst, iso14, iso1430, iso3060, iso6090]])
    predicted_gwl = rf_model.predict(input_data)
    return predicted_gwl[0]

# Example: User inputs (modify with real values)
user_input = {
    'elevation': 100,
    'ndvi': 0.5,
    'lst': 300,
    'iso14': 20,
    'iso1430': 50,
    'iso3060': 30,
    'iso6090': 25
}

# Predict GWL based on the user inputs
predicted_gwl = predict_gwl(**user_input)
print(f"Predicted GWL: {predicted_gwl:.2f}")
