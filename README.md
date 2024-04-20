# ASTATINE (diAbeteS indicaTors classificATIoN modEl)

ASTATINE is a data science project to detect prediabetes and diabetes conditions based on Behavioral Risk Factor Surveillance System (BRFSS) indicators. This project uses diabetes _ 012 _ health _ indicators _ BRFSS2015.csv, a clean dataset of 253,680 survey responses to the CDC's BRFSS2015. The target variable Diabetes_012 has three classes. 0 is for no diabetes or only during pregnancy, 1 is for prediabetes, and 2 is for diabetes. There is a class imbalance in this dataset. This dataset has 21 feature variables. This dataset is from https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data

Before developing a classification model, Exploratory Data Analysis (EDA) and Data Preprocessing are first carried out. The EDA stage is aimed at identifying patterns, finding anomalies, testing hypotheses, and checking assumptions. Data Preprocessing is carried out to eliminate several problems that can interfere with data processing, such as data that is not normally distributed and data imbalance.

The prediction models developed in this project are:
1. SVM
2. XGBoost
3. Random Forest
4. Naive Bayes
5. ANN
