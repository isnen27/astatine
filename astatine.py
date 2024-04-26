# for basic operations
import numpy as np
import pandas as pd
import streamlit as st

#for visualizations
import matplotlib.pyplot as plt
import math
import seaborn as sns
from pandas import plotting
import matplotlib.style as style
style.use("fivethirtyeight")
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid_spec
%matplotlib inline
import missingno as msno

# Extended File
from eda import
from model import
from prepo import




from keras.models import load_model
# Load the saved model
saved_model_path = "ann_model.h5"
loaded_model = load_model(saved_model_path)

# Define the range of values for each column
column_ranges = {
    "HighBP": "(0 for No, 1 for Yes)",
    "HighChol": "(0 for No, 1 for Yes)",
    "BMI": "(Numeric value, based on formula)",
    "Smoker": "(0 for No, 1 for Yes)",
    "Stroke": "(0 for No, 1 for Yes)",
    "HeartDiseaseorAttack": "(0 for No, 1 for Yes)",
    "PhysActivity": "(0 for No, 1 for Yes)",
    "HvyAlcoholConsump": "(0 for No, 1 for Yes)",
    "NoDocbcCost": "(0 for No, 1 for Yes)",
    "GenHlth": "(Range: 1-5)",
    "MentHlth": "(Numeric value between 0-30)",
    "PhysHlth": "(Numeric value between 0-30)",
    "DiffWalk": "(0 for No, 1 for Yes)",
    "Age": "(Range: 1-13)",
    "Education": "(Range: 1-6)",
    "Income": "(Range: 1-8)"
}

# Create UI for input
st.title("Diabetes Prediction App")

new_data = {}
for column in column_ranges:
    new_value = st.text_input(f"{column} {column_ranges[column]}")
    new_data[column] = float(new_value) if new_value else None

# Convert input data to DataFrame
new_df = pd.DataFrame([new_data])

# Make prediction if all values are provided
if all(value is not None for value in new_data.values()):
    # Predict using the loaded model
    predicted_diabetes = loaded_model.predict(new_df)

    # Print prediction result
    if predicted_diabetes[0][0] >= 0.5:
        st.write("Based on our research model, it is predicted that you have a Prediabetes/Diabetes status. We recommend you see a doctor soon.")
    else:
        st.write("Based on our research model, it is predicted that you do not have a Diabetes status. Enjoy your life.")
else:
    st.write("Please provide values for all features to make a prediction.")



