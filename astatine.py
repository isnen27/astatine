# for basic operations
import numpy as np
import pandas as pd
import streamlit as st
import os
import logging

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
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import io

# for model preparation
import sklearn
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, accuracy_score
from mlxtend.plotting import plot_confusion_matrix

#for modelling
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense
import itertools
import tensorflow as tf

#Save model
import joblib
import warnings
import pickle
from keras.models import save_model
from keras.models import load_model
from tensorflow.keras.models import load_model

# Extended File
from eda import describe_detail, html_report, display_column_info,plot_binary_pie,shapiro_test,plot_boxplots, plot_histograms, apply_mappings,plot_categorical_distribution, plot_bmi_distribution, plot_bmi_diabetes_relation, plot_stacked_bar, plot_diabetes_frequency_by_age, plot_education_diabetes_relation, plot_income_diabetes_relation, plot_diabetes_frequency_by_genhlth, plot_correlation_heatmap, plot_correlation_with_target, calc_VIF, perform_ANOVA, perform_ChiSquare
from prepo import imbalance_treatment, imbalance_treatment2, data_split_and_scale
from model import svm_model, plot_confusion_matrix_svm, xgboost_model, plot_confusion_matrix_xg, random_forest_model, plot_confusion_matrix_rf, naive_bayes_model, plot_confusion_matrix_gnb, ann_model, plot_confusion_matrix_ann, evaluate_models, plot_roc_auc


# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
    return df
df = load_data()
# Initialize session state if not already done
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    st.session_state['df'] = df.copy()
    st.session_state['x_sm'] = None
    st.session_state['y_sm'] = None
    st.session_state['X_train'] = None
    st.session_state['X_test'] = None
    st.session_state['Y_train'] = None
    st.session_state['Y_test'] = None
    st.session_state['X_train_scaled'] = None
    st.session_state['X_test_scaled'] = None

def main(df):
    # Session State Handling
    # Main Page Design
    st.title(':stethoscope: :blue[ASTATINE]')
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
       st.write('''ASTATINE is a data science project to detect prediabetes and diabetes conditions based on Behavioral Risk Factor Surveillance System (BRFSS) indicators. This project uses diabetes _ binary _ health _ indicators _ BRFSS2015.csv is a clean dataset of 253,680 survey responses to the CDC's BRFSS2015. The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. This dataset has 21 feature variables and is not balanced. This dataset has 21 feature variables. This dataset is from Kaggle.
    	
Before developing a classification model, Exploratory Data Analysis (EDA) and Data Preprocessing are first carried out. The EDA stage is aimed at identifying patterns, finding anomalies, testing hypotheses, and checking assumptions. Data Preprocessing is carried out to eliminate several problems that can interfere with data processing, such as data that is not normally distributed and data imbalance.
			  
The prediction models developed in this project are:
1. SVM
2. XGBoost
3. Random Forest
4. Naive Bayes
5. ANN''')
	
    if menu == "Dataset Description" and menu2 == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
       expand1 = st.expander("Content")
       expand1.write("The Behavioral Risk Factor Surveillance System (BRFSS) is a health-related telephone survey that is collected annually by the CDC. Each year, the survey collects responses from over 400,000 Americans on health-related risk behaviors, chronic health conditions, and the use of preventative services. It has been conducted every year since 1984. For this project, a csv of the dataset available on Kaggle for the year 2015 was used. This original dataset contains responses from 441,455 individuals and has 330 features. These features are either questions directly asked of participants, or calculated variables based on individual participant responses.")
       expand2 = st.expander("Dataset")
       expand2.write("diabetes _ binary _ health _ indicators _ BRFSS2015.csv is a clean dataset of 253,680 survey responses to the CDC's BRFSS2015. The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. This dataset has 21 feature variables and is not balanced.")
       expand3 = st.expander("About Columns :")
       expand3.write("The dataset originally has **330 features (columns)** but based on diabetes disease research regarding factors influencing diabetes disease and other chronic health conditions the publisher of that dataset clean BRFSS data into **a useable format for machine learning alogrithms.** So, this dataset only consist of **21 features**.")
       display_column_info()

    if menu == "Basic Exploration" and menu2 == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
       st.subheader("Let's begin!")
       medic_option = st.radio("Are you medical staff?", ["Yes", "No"])
       if st.button("Process"):
          if medic_option == "Yes":
             st.title('Diabetes Binary Health Indicators - BRFSS2015')
             describe_detail(df)
          elif medic_option == "No":
             # Display HTML Report
             st.title('Diabetes Binary Health Indicators - BRFSS2015')
             with open("final_project_eda_report.html", "r") as f:
                   html_content = f.read()
             #st.markdown(html_content, unsafe_allow_html=True)
             components.html(html_content, height=600, scrolling=True)
    if menu == "Univariat Analysis & Insight" and menu2 == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
       variabel_biner = ['Diabetes_binary', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke','HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']
       st.subheader("Check Data Distributions for Binary Variables")
       plot_binary_pie(df, variabel_biner)
       st.subheader("Shapiro-Wilk Normality Test")
       selected_column = st.selectbox("Select a column to perform normality test:", df.columns)
       alpha = st.slider("Select significance level (alpha):", min_value=0.01, max_value=0.1, step=0.01, value=0.05)
       result = shapiro_test(df[selected_column], alpha)
       st.write(f"Shapiro-Wilk Test for {selected_column}:")
       st.write(result)
       cols = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
       st.subheader("Check Data Distributions with Boxplots")
       plot_boxplots(df, cols)
       st.write('''Conclusion: There's outlier in column BMI, GenHlth, MenHlth, PhysHlth, but all columns will transpose to categorical value.''')
       st.subheader("Check and Drop Duplicated Data")
       st.write("Duplicated data :", df.duplicated().sum())
       df.drop_duplicates()
       st.write("Data shape after remove duplicated data:", df.shape)
       columns_to_convert = ["Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack","PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
       df[columns_to_convert] = df[columns_to_convert].astype(int)
       st.subheader("Histogram")
       st.set_option('deprecation.showPyplotGlobalUse', False)
       plot_histograms(df)
       st.subheader("Insight from Data")
       df_1 = df.copy()
       df_2 = df.copy()
       df_2 = apply_mappings(df_2)
       st.subheader("Categorical Variable Distributions")
       plot_categorical_distribution(df_2)
       st.write('''
       1.For the Age variable, the highest frequency is found in the age groups of 60 to 64 and 65 to 69. Most of the data is distributed among age groups that are no longer productive.\n 
       2.The distribution of the General Health (GenHlth) feature is dominated by the categories Fair and Good. Generally, respondents show an intermediate level of general health, not bad, but not in prime condition either.\n 
       3.For the Mental Health (MentHlth) feature, the distribution is not normal with a tendency towards being positively skewed. Most respondents exhibit good mental health levels.\n 
       4.Regarding the Physical Activity (PhysActivity) feature, some respondents do not allocate time for physical activities outside their routine activities. In other words, the majority of respondents do not exercise regularly.\n 
       5.Based on the frequency distribution of the Education feature, most respondents are educated individuals as they have completed higher education levels.\n 
       6.Based on the frequency distribution of the Income feature, most respondents are high-income individuals ($75,000 or more).\n''')
       st.subheader("BMI Category Distribution")
       plot_bmi_distribution(df_2)
       st.write("Based on the distribution of BMI, the majority of respondents are experiencing Obesity and Overweight. The selection of respondents for the sample is appropriate considering that both categories are indeed prone to degenerative diseases, one of which is diabetes.")
       st.subheader("Relation between BMI and Diabetes")
       plot_bmi_diabetes_relation(df_1)
       st.write("Based on the BMI distribution graph, BMI values experiencing prediabetes/diabetes conditions range between 20 - 40.")
       st.subheader("Stacked Bar Chart")
       cols = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack","PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
       plot_stacked_bar(df_2, cols)
       st.write("The number of healthy people is bigger than people with diabetes")
       st.subheader("Frequency of Diabetes Disease by Age")
       plot_diabetes_frequency_by_age(df_2)
       st.write("As the age increases, the chances of diabetes also commonly increases")
       st.subheader("Relation between Education and Diabetes")
       plot_education_diabetes_relation(df)
       st.write("Most people have a high level of education, and those with higher levels of education tend to experience better overall health.")
       st.subheader("Relation between Income and Diabetes")
       plot_income_diabetes_relation(df)
       st.write("The higher the income, the healthier people become.")
       st.subheader("Frequency of Diabetes Disease by General Health")
       plot_diabetes_frequency_by_genhlth(df_2)
       st.write("The prediabetes/diabetes condition mostly occurs in the Good category. Interestingly, even in the Very Good category, there are still quite a few respondents experiencing prediabetes/diabetes conditions.")
    if menu == "Bivariat Analysis - Feature Selection" and menu2 == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
       st.set_option('deprecation.showPyplotGlobalUse', False)
       df.drop_duplicates()
       columns_to_convert = ["Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack","PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]
       df[columns_to_convert] = df[columns_to_convert].astype(int)
       st.subheader("Correlation of Features")
       plot_correlation_heatmap(df)
       st.subheader("Correlation with Diabetes_binary")
       plot_correlation_with_target(df)
       st.write('''Conclusion:\n
Strong Correlation: GenHlth, HighBP, DiffWalk, BMI\n
Moderate Corellation with positive relation: HighChol, Age, HeartDiseaseorAttack, PhysHlth, Stroke, CholCheck, MentHlth\n
Moderate Corellation with negative relation: Income, Education, PhysActivity, HvyAlcoholConsump\n
Weak Correlation: Smoker, Sex, AnyHealthcare, NoDocbcCost, Fruits, Veggies''')
       st.subheader("VIF (Variance Inflation Factor)")
       vif_result = calc_VIF(df)
       st.write(vif_result)
       st.subheader("ANOVA Test")
       anova_result = perform_ANOVA(df)
       st.write(anova_result)
       st.subheader("Chi-Square Test")
       chi_square_result = perform_ChiSquare(df)
       st.write(chi_square_result)
       st.write('''We will drop column "Fruits" , "Veggies" , "NoDocbcCost" , "CholCheck" , " AnyHealthcare"''')
       colomns = ["Fruits" , "Veggies" , "NoDocbcCost" , "CholCheck" , "AnyHealthcare"]
       df = df.drop(colomns , axis= 1)
       st.write("Statistics Descriptive Dataframe after Feature Selection :")
       st.write(df.describe().T)
    if menu2 == "Treatment for imbalance dataset" and menu == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
       if st.session_state['x_sm'] is None or st.session_state['y_sm'] is None:
          st.session_state['x_sm'], st.session_state['y_sm'] = imbalance_treatment(st.session_state['df'])
       st.write("Y-value shape :", st.session_state['y_sm'].shape)
       st.write("X-value shape :", st.session_state['x_sm'].shape)
       st.write("Y-value count after resampling:", st.session_state['y_sm'].value_counts())
       st.write("Descriptive statistics after resampling:", st.session_state['df'].describe().T)
    if menu2 == "Data Splitting & Data Scalling" and menu == "- - - - -" and menu3 == "- - - - -" and menu4 == "- - - - -":
       if st.session_state['X_train'] is None or st.session_state['X_test'] is None:
          if st.session_state['x_sm'] is None or st.session_state['y_sm'] is None:
             st.session_state['x_sm'], st.session_state['y_sm'] = imbalance_treatment2(st.session_state['df'])
          st.session_state['X_train'], st.session_state['X_test'], st.session_state['Y_train'], st.session_state['Y_test'] = train_test_split(st.session_state['x_sm'], st.session_state['y_sm'], test_size=0.3, random_state=37)
          scaler = StandardScaler()
          st.session_state['X_train_scaled'] = scaler.fit_transform(st.session_state['X_train'])
          st.session_state['X_test_scaled'] = scaler.fit_transform(st.session_state['X_test'])
       st.write("X train data:", st.session_state['X_train_scaled'])
       st.write("X test data:", st.session_state['X_test_scaled'])
    if menu3 == "Machine Learning Model" and menu == "- - - - -" and menu2 == "- - - - -" and menu4 == "- - - - -":
       if st.session_state['X_train_scaled'] is None or st.session_state['X_test_scaled'] is None:
          if st.session_state['x_sm'] is None or st.session_state['y_sm'] is None:
             st.session_state['x_sm'], st.session_state['y_sm'] = imbalance_treatment2(st.session_state['df'])
          st.session_state['X_train'], st.session_state['X_test'], st.session_state['Y_train'], st.session_state['Y_test'] = train_test_split(st.session_state['x_sm'], st.session_state['y_sm'], test_size=0.3, random_state=37)
          scaler = StandardScaler()
          st.session_state['X_train_scaled'] = scaler.fit_transform(st.session_state['X_train'])
          st.session_state['X_test_scaled'] = scaler.fit_transform(st.session_state['X_test'])
       # Create an expander for SVM model
       st.subheader("Machine Learning Model")
       with st.expander("SVM Model"):
            svm_model(st.session_state['X_train_scaled'], st.session_state['Y_train'], st.session_state['X_test_scaled'], st.session_state['Y_test'])
       with st.expander("XGBoost Model"):
            xgboost_model(st.session_state['X_train_scaled'], st.session_state['Y_train'], st.session_state['X_test_scaled'], st.session_state['Y_test'])
       with st.expander("Random Forest Model"):
            random_forest_model(st.session_state['X_train_scaled'], st.session_state['Y_train'], st.session_state['X_test_scaled'], st.session_state['Y_test'])
       with st.expander("Naive Bayes Model"):
            naive_bayes_model(st.session_state['X_train_scaled'], st.session_state['Y_train'], st.session_state['X_test_scaled'], st.session_state['Y_test'])
       with st.expander("ANN Model"):
            ann_model(st.session_state['X_train_scaled'], st.session_state['Y_train'], st.session_state['X_test_scaled'], st.session_state['Y_test'])
        
    if menu3 == "Model Evaluation" and menu == "- - - - -" and menu2 == "- - - - -" and menu4 == "- - - - -":
       if st.session_state['X_train_scaled'] is None or st.session_state['X_test_scaled'] is None:
          if st.session_state['x_sm'] is None or st.session_state['y_sm'] is None:
             st.session_state['x_sm'], st.session_state['y_sm'] = imbalance_treatment2(st.session_state['df'])
          st.session_state['X_train'], st.session_state['X_test'], st.session_state['Y_train'], st.session_state['Y_test'] = train_test_split(st.session_state['x_sm'], st.session_state['y_sm'], test_size=0.3, random_state=37)
          scaler = StandardScaler()
          st.session_state['X_train_scaled'] = scaler.fit_transform(st.session_state['X_train'])
          st.session_state['X_test_scaled'] = scaler.fit_transform(st.session_state['X_test'])
       st.subheader("Model Evaluation")
       #SVM
       svm = SVC(kernel='rbf', C=1.0)
       svm.fit(st.session_state['X_train_scaled'], st.session_state['Y_train'])
       y_pred_svm = svm.predict(st.session_state['X_test_scaled'])
       train_score_svm = svm.score(st.session_state['X_train_scaled'], st.session_state['Y_train'])
       test_score_svm = svm.score(st.session_state['X_test_scaled'], st.session_state['Y_test'])
       mse_svm = mean_squared_error(st.session_state['Y_test'], y_pred_svm)
       rmse_svm = np.sqrt(mse_svm)
       
       #XGBost
       xg = XGBClassifier(eval_metric='error', learning_rate=0.1)
       xg.fit(st.session_state['X_train_scaled'], st.session_state['Y_train'])
       y_pred_xg = xg.predict(st.session_state['X_test_scaled'])
       train_score_xg = xg.score(st.session_state['X_train_scaled'], st.session_state['Y_train'])
       test_score_xg = xg.score(st.session_state['X_test_scaled'], st.session_state['Y_test'])
       mse_xg = mean_squared_error(st.session_state['Y_test'], y_pred_xg)
       rmse_xg = np.sqrt(mse_xg)
       
       #Random Forest
       rf = RandomForestClassifier(max_depth=12, n_estimators=10, random_state=42)
       rf.fit(st.session_state['X_train_scaled'], st.session_state['Y_train'])
       y_pred_rf = rf.predict(st.session_state['X_test_scaled'])
       train_score_rf = rf.score(st.session_state['X_train_scaled'], st.session_state['Y_train'])
       test_score_rf = rf.score(st.session_state['X_test_scaled'], st.session_state['Y_test'])
       mse_rf = mean_squared_error(st.session_state['Y_test'], y_pred_rf)
       rmse_rf = np.sqrt(mse_rf)
       
       #Naive Bayes
       gnb = GaussianNB()
       gnb.fit(st.session_state['X_train_scaled'], st.session_state['Y_train'])
       y_pred_gnb = gnb.predict(st.session_state['X_test_scaled'])
       train_score_gnb = gnb.score(st.session_state['X_train_scaled'], st.session_state['Y_train'])
       test_score_gnb = gnb.score(st.session_state['X_test_scaled'], st.session_state['Y_test'])
       mse_gnb = mean_squared_error(st.session_state['Y_test'], y_pred_gnb)
       rmse_gnb = np.sqrt(mse_gnb)
       
       #ANN
       model_ann = Sequential()
       model_ann.add(Dense(64, activation='relu', input_dim=st.session_state['X_train_scaled'].shape[1]))
       model_ann.add(Dense(64, activation='relu'))
       model_ann.add(Dense(1, activation='sigmoid'))
       model_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
       history = model_ann.fit(st.session_state['X_train_scaled'], st.session_state['Y_train'], epochs=10, batch_size=32, validation_data=(st.session_state['X_test_scaled'], st.session_state['Y_test']), verbose=0)
       y_pred_prob_ann = model_ann.predict(st.session_state['X_test_scaled'])
       y_pred_ann = (y_pred_prob_ann > 0.5).astype(int)
       train_score_ann = model_ann.evaluate(st.session_state['X_train_scaled'], st.session_state['Y_train'], verbose=0)
       test_score_ann = model_ann.evaluate(st.session_state['X_test_scaled'], st.session_state['Y_test'], verbose=0)
       mse_ann = mean_squared_error(st.session_state['Y_test'], y_pred_ann)
       rmse_ann = np.sqrt(mse_ann)

       evaluated_models = evaluate_models(
	       st.session_state['X_train_scaled'], st.session_state['X_test_scaled'], st.session_state['Y_train'], st.session_state['Y_test'],
	       svm, xg, rf, gnb, train_score_ann, test_score_ann,mse_svm, mse_xg, mse_rf, mse_gnb, mse_ann,rmse_svm, rmse_xg, rmse_rf, rmse_gnb, rmse_ann)
       st.dataframe(evaluated_models)
       st.subheader("ROC Plot")
       plot_roc_auc(
       st.session_state['X_test_scaled'], st.session_state['Y_test'],y_pred_svm, y_pred_xg, y_pred_rf, y_pred_gnb, y_pred_ann)
              
    if menu4 == "ASTATINE App" and menu == "- - - - -" and menu2 == "- - - - -" and menu3 == "- - - - -":
       # Load the saved model
       saved_model_path = 'ann_model.keras'
       loaded_model = load_model(saved_model_path)
       if not os.path.exists(saved_model_path):
              st.error(f"Model file does not exist at the specified path: {saved_model_path}")
       else:
           try:
              loaded_model = tf.keras.models.load_model(saved_model_path)
              st.success("Model loaded successfully")
           except Exception as e:
              logging.error("Error loading model", exc_info=True)
              st.error("Failed to load the model. Please check the logs for more details.")
              return
       # Define the range of values for each column
       column_ranges = {
        "HighBP": ["No", "Yes"],
        "HighChol": ["No", "Yes"],
        "BMI": "Numeric value, based on formula",
        "Smoker": ["No", "Yes"],
        "Stroke": ["No", "Yes"],
        "HeartDiseaseorAttack": ["No", "Yes"],
        "PhysActivity": ["No", "Yes"],
        "HvyAlcoholConsump": ["No", "Yes"],
        "Sex": ["Female", "Male"],
        "GenHlth": list(range(1, 6)),
        "MentHlth": (0, 30),
        "PhysHlth": (0, 30),
        "DiffWalk": ["No", "Yes"],
        "Age": list(range(1, 14)),
        "Education": list(range(1, 7)),
        "Income": list(range(1, 9))
    }
       # Create UI for input
       st.title("Diabetes Prediction App")
       expand_pred = st.expander("Filling instructions : ")
       expand_pred.write('''1. For variables with options 0 and 1, 0 stands for Yes and 1 stands for No. Except for Sex, where 0 stands for Female and 1 stands for Male.\n
2. For the MentHlth and PhysHlth variables, values can be entered within the range of 0 - 30. The number indicates the number of days experiencing disturbances in mental or physical health.\n 
3. For entries in the GenHlth variable, indicate the general health level as follows:\n
\t1: 'Poor'\n
\t2: 'Fair'\n
\t3: 'Good'\n
\t4: 'Very Good'\n
\t5: 'Excellent'\n
4. For Education, categories can be selected from 1 - 6 as follows:\n
\t1: 'Never Attended School'\n
\t2: 'Elementary'\n
\t3: 'Junior High School'\n
\t4: 'Senior High School'\n
\t5: 'Some college or technical school'\n
\t6: 'College graduate'\n
5. For Income, categories can be selected from 1 - 8 as follows:\n
\t1: 'Less than $10,000'\n
\t2: '$10,000 to $14,999'\n
\t3: '$15,000 to $19,999'\n
\t4: '$20,000 to $24,999'\n
\t5: '$25,000 to $34,999'\n
\t6: '$35,000 to $49,999'\n
\t7: '$50,000 to $74,999'\n
\t8: '$75,000 or More'\n
6. For Age, categories can be selected from 1 - 13 as follows:\n
\t1: '18 to 24'\n
\t2: '25 to 29'\n
\t3: '30 to 34'\n
\t4: '35 to 39'\n
\t5: '40 to 44'\n
\t6: '45 to 49'\n
\t7: '50 to 54'\n
\t8: '55 to 59'\n
\t9: '60 to 64'\n
\t10: '65 to 69'\n
\t11: '70 to 74'\n
\t12: '75 to 79'\n
\t13: '80 or older''')
       
       new_data = {}
       for column, options in column_ranges.items():
           if isinstance(options, list):  # Radio button
              new_value = st.radio(f"{column} {options}", options)
           elif isinstance(options, tuple):  # Slider
              new_value = st.slider(f"{column} ({options[0]}-{options[1]})", options[0], options[1])
           elif column == "BMI":  # Numeric input
              new_value = st.number_input(f"{column} ({options})", min_value=0.0)
           else:  # Selectbox
              new_value = st.selectbox(f"{column}", options)
           if column == "Sex":  # Convert "Male" or "Female" to numeric
              new_value = 0 if new_value == "Female" else 1
           if new_value == "No":
              new_value = 0
           elif new_value == "Yes":
              new_value = 1
           new_data[column] = new_value

       # Process button
       if st.button("Process"):
          # Convert input data to DataFrame
          new_df = pd.DataFrame([new_data])

          # Make prediction if all values are provided
          if all(value is not None for value in new_data.values()):
               # Predict using the loaded model
               predicted_diabetes = loaded_model.predict(new_df)
               # Print prediction result
               if predicted_diabetes[0] > 0.5:
                  st.write("Based on our research model, it is predicted that you have a Prediabetes/Diabetes status. We recommend you see a doctor soon.")
               else:
                  st.write("Based on our research model, it is predicted that you do not have a Diabetes status. Enjoy your life.")
          else:
            st.write("Please provide values for all features to make a prediction.")

    if menu4 == "Conclution Remark" and menu == "- - - - -" and menu2 == "- - - - -" and menu3 == "- - - - -":
       st.subheader("Conclution Remark")
       st.write('''
       1. The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. This dataset has 21 feature variables and is not balanced. This dataset has 21 feature variables.\n
       2. Variables that has strong corralation with diabetes are general health (GenHlth), high blood pressure (HighBP), difficulty walking (DiffWalk), and Body Mass Index (BMI).\n
       3. The best model for predict prediabetes/diabetes stage is **ANN** with **accuracy of 0.87**.\n''')


if __name__=="__main__":
    df = load_data()
    main(df)
