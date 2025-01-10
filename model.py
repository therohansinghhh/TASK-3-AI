import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model():
    data = pd.read_csv('car_details_v4.csv')

    # Label encode categorical columns
    label_encoder = LabelEncoder()
    data['Fuel Type'] = label_encoder.fit_transform(data['Fuel Type'])
    data['Transmission'] = label_encoder.fit_transform(data['Transmission'])

    # Feature selection
    features = ['Make', 'Model', 'Year', 'Kilometer', 'Fuel Type', 'Transmission', 'Engine', 'Max Power', 'Max Torque', 'Drivetrain', 'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity']
    X = data[features]
    y = data['Price']

    # Handle Engine data (strip 'cc' and convert to numeric)
    X['Engine'] = X['Engine'].str.replace(r'[^0-9]', '', regex=True).astype(float)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'car_price_model.pkl')

if __name__ == '__main__':
    train_model()