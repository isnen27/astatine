# ASTATINE (diAbeteS indicaTors classificATIoN modEl)

ASTATINE is a data science project to detect prediabetes and diabetes conditions based on Behavioral Risk Factor Surveillance System (BRFSS) indicators. This project uses diabetes _ binary _ health _ indicators _ BRFSS2015.csv is a clean dataset of 253,680 survey responses to the CDC's BRFSS2015. The target variable Diabetes_binary has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. This dataset has 21 feature variables and is not balanced. This dataset has 21 feature variables. This dataset is from https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?resource=download&select=diabetes_binary_health_indicators_BRFSS2015.csv

Before developing a classification model, Exploratory Data Analysis (EDA) and Data Preprocessing are first carried out. The EDA stage is aimed at identifying patterns, finding anomalies, testing hypotheses, and checking assumptions. Data Preprocessing is carried out to eliminate several problems that can interfere with data processing, such as data that is not normally distributed and data imbalance.

The prediction models developed in this project are:
1. SVM
2. XGBoost
3. Random Forest
4. Naive Bayes
5. ANN
