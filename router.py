# Import the Libraries
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request
import joblib
import os
# the function I craeted to process the data in utils.py
from utils import preprocess_new


# Intialize the Flask APP
app = Flask(__name__)

# Loading the Model
model = joblib.load('model_XGBoost.pkl')

# Route for Home page


@app.route('/')
def home():
    return render_template('index.html')

# Route for Predict page


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':  # while prediction
        long = float(request.form['long'])
        latit = float(request.form['latit'])
        med_age = float(request.form['med_age'])
        total_rooms = float(request.form['total_rooms'])
        total_bedrooms = float(request.form['total_bedrooms'])
        pop = float(request.form['pop'])
        hold = float(request.form['hold'])
        income = float(request.form['income'])
        ocean = request.form['ocean']

        # Remmber the Feature Engineering we did
        rooms_per_hold = total_rooms / hold
        bedroms_per_rooms = total_bedrooms / total_rooms
        pop_per_hold = pop / hold

        # Concatenate all Inputs
        X_new = pd.DataFrame({'longitude': [long], 'latitude': [latit], 'housing_median_age': [med_age], 'total_rooms': [total_rooms],
                              'total_bedrooms': [total_bedrooms], 'population': [pop], 'households': [hold], 'median_income': [income],
                              'ocean_proximity': [ocean], 'rooms_per_household': [rooms_per_hold], 'bedroms_per_rooms': bedroms_per_rooms,
                              'population_per_household': [pop_per_hold]
                              })

        # Call the Function and Preprocess the New Instances
        X_processed = preprocess_new(X_new)

        # call the Model and predict
        y_pred_new = model.predict(X_processed)
        y_pred_new = '{:.4f}'.format(y_pred_new[0])

        return render_template('predict.html', pred_val=y_pred_new)
    else:
        return render_template('predict.html')


# Route for About page
@app.route('/about')
def about():
    return render_template('about.html')


# Run the App from the Terminal
if __name__ == '__main__':
    app.run(debug=True)
