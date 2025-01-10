from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset
data = pd.read_csv(r'C:/Users/Pranav/Desktop/Compozent Tasks/ML project/car details v4.csv')

# Function to extract numeric values from strings like '184 bhp'
def extract_numeric_value(text):
    if isinstance(text, str):
        match = re.findall(r'\d+', text)
        if match:
            return int(match[0])  # Return the first numeric value found
    return np.nan  # Return NaN if the value is not a string or doesn't contain numbers

# Clean 'Engine' column to remove non-numeric characters (e.g., '1197 cc' -> 1197)
data['Engine'] = data['Engine'].apply(lambda x: re.sub(r'\D', '', str(x)))
data['Engine'] = pd.to_numeric(data['Engine'], errors='coerce')

# Clean 'Max Power' and 'Max Torque' columns to extract only numeric values
data['Max Power'] = data['Max Power'].apply(extract_numeric_value)
data['Max Torque'] = data['Max Torque'].apply(extract_numeric_value)

# Select relevant features for the model
features = ['Year', 'Kilometer', 'Engine', 'Max Power', 'Max Torque', 'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity']
target = 'Price'

# Fill missing values only for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Fill missing values for non-numeric columns (optional, can be done with mode or a default value)
data.fillna(data.mode().iloc[0], inplace=True)

# Prepare the data for training
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model each time the app runs (no need for .pkl file)
model = LinearRegression()
model.fit(X_train, y_train)

# Route for home page (GET method)
@app.route('/')
def home():
    return render_template('index.html', prediction=None)

# Route for predictions (POST method)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data from user
        make = request.form['make']
        model_input = request.form['model']
        year = int(request.form['year'])
        kilometer = float(request.form['kilometer'])
        fuel_type = request.form['fuel_type']
        transmission = request.form['transmission']
        engine = request.form['engine']
        max_power = request.form['max_power']
        max_torque = request.form['max_torque']
        drivetrain = request.form['drivetrain']
        length = float(request.form['length'])
        width = float(request.form['width'])
        height = float(request.form['height'])
        seating_capacity = int(request.form['seating_capacity'])
        fuel_tank_capacity = float(request.form['fuel_tank_capacity'])

        # Extract numeric values from Max Power and Max Torque
        max_power = extract_numeric_value(max_power)
        max_torque = extract_numeric_value(max_torque)

        # Clean the engine input (remove non-numeric characters)
        engine = re.sub(r'\D', '', engine)
        engine = int(engine)  # Convert to integer

        # Prepare the input data for prediction
        input_data = np.array([[year, kilometer, engine, max_power, max_torque, length, width, height, seating_capacity, fuel_tank_capacity]])

        # Predict the price
        predicted_price = model.predict(input_data)[0]

        # Ensure the price is positive
        predicted_price = abs(predicted_price)

        # Return to home page with the predicted price
        return render_template('index.html', prediction=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)