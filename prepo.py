import streamlit as st
import pandas as pd

# for model preparation
import sklearn
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

def imbalance_treatment(df):
    df = df.drop_duplicates()
    columns_to_drop = ["Fruits", "Veggies", "Sex", "CholCheck", "AnyHealthcare"]
    df = df.drop(columns_to_drop, axis=1)
    X = df.drop("Diabetes_binary", axis=1)
    Y = df["Diabetes_binary"]
    
    st.subheader("Treatment for imbalance dataset")
    st.write("Y-value count :", Y.value_counts())
    
    # Applying NearMiss for under-sampling
    nm = NearMiss(version=1, n_neighbors=10)
    x_sm, y_sm = nm.fit_resample(X, Y)
    
    return x_sm, y_sm

def imbalance_treatment2(df):
    df = df.drop_duplicates()
    columns_to_drop = ["Fruits", "Veggies", "Sex", "CholCheck", "AnyHealthcare"]
    df = df.drop(columns_to_drop, axis=1)
    X = df.drop("Diabetes_binary", axis=1)
    Y = df["Diabetes_binary"]
    
    # Applying NearMiss for under-sampling
    nm = NearMiss(version=1, n_neighbors=10)
    x_sm, y_sm = nm.fit_resample(X, Y)
    
    return x_sm, y_sm

def data_split_and_scale(x_sm, y_sm):
    # Data split
    X_train, X_test, Y_train, Y_test = train_test_split(x_sm, y_sm, test_size=0.3, random_state=37)
    
    # Feature scaling
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)
    
    return X_train_scaled, X_test_scaled, Y_train, Y_test
