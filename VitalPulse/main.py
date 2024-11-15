#heart working----------------------------------------------------------------------
# Importing the libraries
import numpy as np
import pandas as pd
import random

dataset = pd.read_csv('./heart.csv')
dataset['Sex'] = dataset['Sex'].apply(lambda x: 0 if x == 'M' else 1)
dataset['ExerciseAngina'] = dataset['ExerciseAngina'].apply(lambda x: 0 if x == 'N' else 1)
# Create a mapping dictionary
chest_pain_mapping = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
# Map the values using the dictionary
dataset['ChestPainType'] = dataset['ChestPainType'].map(chest_pain_mapping)
X = dataset.iloc[:, :-1].copy()
y = dataset.iloc[:, -1].values

heart_columns = X.columns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6,10])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

# Fit the model to the training data
random_forest_classifier.fit(X_train, y_train)

# Predict the test set results
y_pred_random_forest = random_forest_classifier.predict(X_test)



# bodyfat working-------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('./bodyfat.csv')
X = dataset.iloc[:, [0] + list(range(2, len(dataset.columns)))].copy()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Random Forest Regression
regressor_rf = RandomForestRegressor(n_estimators=10, random_state=0)
regressor_rf.fit(X_train, y_train)

# You can use y_pred_rf directly in the function
body_fat_feature_names = list(X.columns)

# insurance working----------------------------------------------------------------------------------------
# Importing the libraries
import numpy as np
import pandas as pd
# from typing import List, Union
# from fastapi import FastAPI, HTTPException

# Importing the dataset

dataset = pd.read_csv('./expenses.csv')
dataset['sex'] = dataset['sex'].apply(lambda x: 0 if x == 'male' else 1)
dataset['smoker'] = dataset['smoker'].apply(lambda x: 0 if x == 'no' else 1)
X = dataset.iloc[:, :5].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost Regression
from xgboost import XGBRegressor
regressor = XGBRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train_scaled, y_train)

# Assuming you know the actual feature names of your dataset
insurance_feature_names = ['age', 'sex', 'bmi', 'children', 'smoker']

# flask app ---------------------------------------------------------

from flask import Flask, render_template, request

app = Flask(  # Create a flask app
  __name__,
  template_folder='templates',  # Name of html file folder
  static_folder='static'  # Name of directory for static files
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/aboutme')
def aboutme():
    return render_template('aboutme.html')

@app.route('/insurancepredictor')
def insurance():
    return render_template('insurance.html')

@app.route('/fatmetrics')
def fatmetrics():
    return render_template('fatmetrics.html')

@app.route('/heartguardpredictor')
def heartguardpredictor():
    return render_template('heartguardpredictor.html')

# route for handling post request of HEARTGUARD page
@app.route('/process3', methods=['POST'])
def process3():
    try:
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        chest_pain_type = int(request.form['chestPainType'])
        resting_bp = int(request.form['restingBP'])
        cholesterol = int(request.form['cholesterol'])
        fasting_bs = int(request.form['fastingBS'])
        resting_ecg = request.form['restingECG']
        max_hr = int(request.form['maxHR'])
        oldpeak = float(request.form['oldpeak'])
        st_slope = request.form['stSlope']
        exercise_angina = int(request.form['exerciseAngina'])
        values = np.array([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])
        values = pd.DataFrame(values, columns=heart_columns)
        values = np.array(ct.transform(values))

        values = sc.transform(values)

        pred = random_forest_classifier.predict(values)
        # Predict the probabilities for each class
        probabilities = random_forest_classifier.predict_proba(values)
        if(pred[0] == 0):
            display = "There is a {}% likelihood of a potential heart health concern.".format(round(max(probabilities[0]*100)), 2)
        else:
            display = "The analysis indicates a {}% probability of NOT having a heart-related issue.".format(round(max(probabilities[0]*100)), 2)

        return display

    except Exception as e:
        print(f"Error processing form data: {str(e)}")
        return "Internal Server Error", 500

# route for handling post request of BODYFAT page
@app.route('/process2', methods=['POST'])
def process2():
    try:
        density = float(request.form['density'])
        age = int(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        neck = float(request.form['neck'])
        chest = float(request.form['chest'])
        abdomen = float(request.form['abdomen'])
        hip = float(request.form['hip'])
        thigh = float(request.form['thigh'])
        knee = float(request.form['knee'])
        ankle = float(request.form['ankle'])
        biceps = float(request.form['biceps'])
        forearm = float(request.form['forearm'])
        wrist = float(request.form['wrist'])
        values_list = [density, age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist]
        
        y_pred_rf = regressor_rf.predict(pd.DataFrame([values_list], columns=body_fat_feature_names))
    
        response = float(y_pred_rf[0])
        display = "Your body exhibits a body fat percentage of {}%.".format(round(response, 2))
        return display

    except Exception as e:
        print(f"Error processing form data: {str(e)}")
        return "Internal Server Error", 500

#route for handling post request of insurance page

@app.route('/process', methods=['POST'])
def process():
    try:
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoking_status = request.form['smokingStatus']
        sex = 0 if sex == "Male" else 1
        smoking_status = 0 if smoking_status == "No" else 1
        values_list = [age, sex, bmi, children, smoking_status]
        feature_names = insurance_feature_names
        
        y_pred_rf = regressor.predict(pd.DataFrame(scaler.transform([values_list]), columns=feature_names))
        response = float(y_pred_rf[0])
        display = "Over the course of 12 months, your monthly insurance premium amounts to approximately {} INR.".format(round(response, 2))
        return display

    except Exception as e:
        print(f"Error processing form data: {str(e)}")
        return "Internal Server Error", 500

if __name__ == "__main__":  # Makes sure this is the main process
    app.run( # Starts the site
      host='0.0.0.0',  # EStablishes the host, required for repl to detect the site
      port=random.randint(2000, 9000)  # Randomly select the port the machine hosts on.
)
