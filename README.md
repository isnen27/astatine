# ASTATINE (diAbeteS indicaTors classificATIoN modEl)

ASTATINE merupakan proyek data sains untuk mendeteksi kondisi prediabetes dan diabetes berdasarkan indikator Behavioral Risk Factor Surveillance System (BRFSS). This project use diabetes _ 012 _ health _ indicators _ BRFSS2015.csv is a clean dataset of 253,680 survey responses to the CDC's BRFSS2015. The target variable Diabetes_012 has 3 classes. 0 is for no diabetes or only during pregnancy, 1 is for prediabetes, and 2 is for diabetes. There is class imbalance in this dataset. This dataset has 21 feature variables. This dataset from https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data

Sebelum mengembangkan model klasifikasi, terlebih dahulu dilakukan Exploratory Data Analysis (EDA) dan Data Prepocessing. Tahap EDA ditujukan untuk mengidentifikasi pola, menemukan anomali, menguji hipotesis dan memeriksa asumsi. Data Prepocessing dilakukan untuk menghilangkan beberapa permasalahan yang dapat mengganggu saat pemrosesan data, seperti data yang tidak terdistribusi normal dan data imbalance.

Model prediksi yang dikembangkan dalam proyek ini adalah:
1. SVM
2. Decision Tree
3. Random Forest
4. Naive Bayes
5. ANN
