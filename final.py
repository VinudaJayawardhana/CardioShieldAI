
"""
Created on Sun Nov 17 00:13:25 2024

@author: Vinuda
"""
#Loading dataset using pandas library
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer as si 
#Using OneHotEncoder to encode 
from sklearn.preprocessing import OneHotEncoder 
#Using Standardscaler to scale the data
from sklearn.preprocessing import StandardScaler as sc
#Using train_test_split to divide the data into both train and test data 
from sklearn.model_selection import train_test_split
#Uisng linear model to do linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
#To convert fraction to float
from fractions import Fraction
#Using Label Encoding
from sklearn.preprocessing import LabelEncoder

import pickle

# Load the dataset
heart_data = pd.read_csv('heart.csv')
#Applying and calling functions
label_encoder = LabelEncoder()
scaler = sc()
encoder = OneHotEncoder(sparse_output=False)

# Handling MCAR data
heart_data.drop(columns=['PatientID', 'Continent', 'Country', 'Hemisphere'], inplace=True)

# Function to convert fractional string to float
def fraction_to_float(fraction_str):
    try: 
        return float(Fraction(fraction_str)) 
    except ValueError: 
        return None

# Converting Blood Pressure to float values
heart_data['BloodPressure_Float'] = heart_data['BloodPressure'].apply(fraction_to_float)

#  drop the 'BloodPressure' column
heart_data.drop(columns=['BloodPressure'], inplace=True)

# Scaling data
heart_data[['ExerciseHours', 'SedentaryHours', 'BMI']] = scaler.fit_transform(heart_data[['ExerciseHours', 'SedentaryHours', 'BMI']])

# Encoding 'Sex' column
encoded_sex = encoder.fit_transform(heart_data[['Sex']])
encoded_sex_df = pd.DataFrame(encoded_sex, columns=encoder.get_feature_names_out(['Sex']))

# Concatenate the encoded data with the original DataFrame
heart_data = pd.concat([heart_data, encoded_sex_df], axis=1)
heart_data.drop(columns=['Sex'], inplace=True)

# Encoding 'Diet' column using label encoding
heart_data['Diet_encoded'] = label_encoder.fit_transform(heart_data['Diet'])

# Dropping original Diet column
heart_data.drop(columns=['Diet'], inplace=True)


#Training model is creating from here (Regression Model)
#Assigning x as the independent variable 
x = heart_data.drop('HeartAttackRisk', axis=1)
#Assigning y as the dependent variable
y = heart_data['HeartAttackRisk']

#  splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fitting the Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict and evaluate the model
predictedHeartRisk = model.predict(x_test)
mse = mean_squared_error(y_test, predictedHeartRisk)
print("Mean Squared Error:", mse)
'''
def provide_recommendations(features):
    recommendations=[]
    if features['BloodPressure_Float']>120:
        recommendations.append("Maintain healthy blood pressure through diet,exercise and medication.")
    if features['BMI']>25:
        recommendations.append("Consider a balanced diet and regular physica; activity to manage weight")
    if features['ExerciseHours']<2:
        recommendations.append("Increase physical activity to at least 150 minutes per week.")
    if 'Cholesterol'in features and features['Cholesterol'] >200:
         recommendations.append("Monitor your cholesterol levels and consider dietary changes and mediaction.")
    return recommendations


example_features=x_test.iloc[0]
recommendations=provide_recommendations(example_features)
print("Recommendations: ",recommendations)
'''

    
with open(r'C:/Users/Vinuda/OneDrive/Desktop/Heart Disease AI/Heart Disease AI/linear_reg_model.pkl',"wb") as file:
    pickle.dump(model,file)

    print("Model saved to file successfully.")
 