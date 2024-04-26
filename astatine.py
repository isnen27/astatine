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
import missingno as msno
from pandas_profiling import ProfileReport

# Extended File
from eda import describe_detail, html_report
from model import svm_model, plot_confusion_matrix_svm, xgboost_model, plot_confusion_matrix_xg, random_forest_model, plot_confusion_matrix_rf, naive_bayes_model, plot_confusion_matrix_gnb, ann_model, plot_confusion_matrix_ann, evaluate_models, plot_roc_curve
#from prepo import

def main():
    # Session State Handling
    
    # Main Page Design
    st.title(':bookmark_tabs: :blue[ASTATINE]')
    st.header('_:blue[diAbeteS indicaTors classificATIoN modEl]_')
    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Exploratory Data Analysis :", ["- - - - -", 
                                                          "Dataset Description", 
                                                          "Basic Exploration",
                                                          "Univariat Analysis & Insight",
                                                          "Bivariat Analysis - Feature Selection"])
    
    menu2 = st.sidebar.selectbox("Data Prepocessing:", ["- - - - -", 
                                                     "Treatment for imbalance dataset", 
                                                     "Data Splitting & Data Scalling"])
    
    menu3 = st.sidebar.selectbox("Modelling:", ["- - - - -", 
                                                     "Machine Learning Model", 
                                                     "Model Evaluation"])
    menu4 = st.sidebar.selectbox("Prediction:", ["- - - - -", 
                                                     "ASTATINE App", 
                                                     "Conclution Remark"])
    

    # Menu Functions
    if menu == "- - - - -" and menu2 == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
    	st.write('''ASTATINE is a data science project to detect prediabetes and diabetes conditions based on Behavioral Risk Factor Surveillance System (BRFSS) indicators. This project uses diabetes _ binary _ health _ indicators _ BRFSS2015.csv is a clean dataset of 253,680 survey responses to the CDC's BRFSS2015. The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. This dataset has 21 feature variables and is not balanced. This dataset has 21 feature variables. This dataset is from Kaggle.''')
    	st.write('''Before developing a classification model, Exploratory Data Analysis (EDA) and Data Preprocessing are first carried out. The EDA stage is aimed at identifying patterns, finding anomalies, testing hypotheses, and checking assumptions. Data Preprocessing is carried out to eliminate several problems that can interfere with data processing, such as data that is not normally distributed and data imbalance.''')
    	st.write('''The prediction models developed in this project are:
1. SVM
2. XGBoost
3. Random Forest
4. Naive Bayes
5. ANN''')
    if menu == "Dataset Description" and menu2 == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
    	st.subheader("menu1.1")
    if menu == "Basic Exploration" and menu2 == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
    	st.subheader("menu1.2")
    if menu == "Univariat Analysis & Insight" and menu2 == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
    	st.subheader("menu1.3")
    if menu == "Bivariat Analysis - Feature Selection" and menu2 == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
    	st.subheader("menu1.4")
    if menu2 == "Treatment for imbalance dataset" and menu == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
    	st.subheader("menu2.1")
    if menu2 == "Data Splitting & Data Scalling" and menu == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
    	st.subheader("menu2.2")
    if menu3 == "Machine Learning Model" and menu == "- - - - -" and menu2 == "- - - - -" and menu4 == "- - - - -":
    	st.subheader("menu3.1")
    if menu3 == "Model Evaluation" and menu == "- - - - -" and menu2 == "- - - - -" and menu4 == "- - - - -":
    	st.subheader("menu3.2")
    if menu4 == "ASTATINE App" and menu == "- - - - -" and menu2 == "- - - - -" and menu3 == "- - - - -":
    	st.subheader("menu4.1")
    if menu4 == "Conclution Remark" and menu == "- - - - -" and menu2 == "- - - - -" and menu3 == "- - - - -":
    	st.subheader("menu4.2")


if __name__=="__main__":
    main()
