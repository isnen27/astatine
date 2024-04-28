# for basic operations
import numpy as np
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport
from scipy.stats import shapiro
#from pandas.plotting import parallel_coordinates

#for Feature selection
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant  
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#for visualizations
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def display_column_info():
    col1, col2 = st.columns(2)  # Membagi layar menjadi 2 kolom
    # Diabetes_binary
    with col1:
        with st.expander("Diabetes_binary"):
            st.write('''Do you have diabetes stage?
       0: No
       1: Yes''')
    # HighBP
    with col1:
        with st.expander("HighBP"):
            st.write('''Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional
            0: No
            1: Yes''')

    # HighChol
    with col1:
        with st.expander("HighChol"):
            st.write('''Have you EVER been told by a doctor, nurse or other health professional that your blood cholesterol is high?
            0: No
            1: Yes''')

    # CholCheck
    with col1:
        with st.expander("CholCheck"):
            st.write('''Cholesterol check within past five years
            0: No
            1: Yes''')

    # BMI
    with col1:
        with st.expander("BMI"):
            st.write('''Body Mass Index (BMI)''')

    # Smoker
    with col1:
        with st.expander("Smoker"):
            st.write('''Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]
            0: No
            1: Yes''')

    # Stroke
    with col1:
        with st.expander("Stroke"):
            st.write('''(Ever told) you had a stroke.
            0: No
            1: Yes''')

    # HeartDiseaseorAttack
    with col1:
        with st.expander("HeartDiseaseorAttack"):
            st.write('''Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI)
            0: No
            1: Yes''')

    # PhysActivity
    with col1:
        with st.expander("PhysActivity"):
            st.write('''Adults who reported doing physical activity or exercise during the past 30 days other than their regular job
            0: No
            1: Yes''')

    # Fruits
    with col1:
        with st.expander("Fruits"):
            st.write('''Consume Fruit 1 or more times per day
            0: No
            1: Yes''')

    # Veggies
    with col1:
        with st.expander("Veggies"):
            st.write('''Consume Vegetables 1 or more times per day
            0: No
            1: Yes''')

    # HvyAlcoholConsump
    with col2:
        with st.expander("HvyAlcoholConsump"):
            st.write('''Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)
            0: No
            1: Yes''')

    # AnyHealthcare
    with col2:
        with st.expander("AnyHealthcare"):
            st.write('''Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service?
            0: No
            1: Yes''')

    # NoDocbcCost
    with col2:
        with st.expander("NoDocbcCost"):
            st.write('''Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?
            0: No
            1: Yes''')

    # GenHlth
    with col2:
        with st.expander("GenHlth"):
            st.write('''Would you say that in general your health is:''')
            st.write("1: Poor")
            st.write("2: Fair")
            st.write("3: Good")
            st.write("4: Very Good")
            st.write("5: Excellent")

    # MentHlth
    with col2:
        with st.expander("MentHlth"):
            st.write('''Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? (0 - 30)''')

    # PhysHlth
    with col2:
        with st.expander("PhysHlth"):
            st.write('''Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (0 - 30)''')

    # DiffWalk
    with col2:
        with st.expander("DiffWalk"):
            st.write('''Do you have serious difficulty walking or climbing stairs?
            0: No
            1: Yes''')

    # Sex
    with col2:
        with st.expander("Sex"):
            st.write('''Indicate sex of respondent
            0: Female
            1: Male''')

    # Age
    with col2:
        with st.expander("Age"):
            st.write('''Fourteen-level age category:''')
            st.write("1: 18 to 24")
            st.write("2: 25 to 29")
            st.write("3: 30 to 34")
            st.write("4: 35 to 39")
            st.write("5: 40 to 44")
            st.write("6: 45 to 49")
            st.write("7: 50 to 54")
            st.write("8: 55 to 59")
            st.write("9: 60 to 64")
            st.write("10: 65 to 69")
            st.write("11: 70 to 74")
            st.write("12: 75 to 79")
            st.write("13: 80 or older")

    # Education
    with col2:
        with st.expander("Education"):
            st.write('''What is the highest grade or year of school you completed?''')
            st.write("1: Never Attended School")
            st.write("2: Elementary")
            st.write("3: Junior High School")
            st.write("4: Senior High School")
            st.write("5: Some college or technical school")
            st.write("6: College graduate")

    # Income
    with col2:
        with st.expander("Income"):
            st.write('''Is your annual household income from all sources:''')
            st.write("1: Less than $10,000")
            st.write("2: $10,000 to $14,999")
            st.write("3: $15,000 to $19,999")
            st.write("4: $20,000 to $24,999")
            st.write("5: $25,000 to $34,999")
            st.write("6: $35,000 to $49,999")
            st.write("7: $50,000 to $74,999")
            st.write("8: $75,000 or More")

def describe_detail(df):
    # (a) First five data points
    st.title('First five data points')
    st.write(df.head())

    # (b) Random five data points
    st.title('Random five data points')
    st.write(df.sample(5))

    # (c) Last five data points
    st.title('Last five data points')
    st.write(df.tail())

    # (d) Shape and Size of data set
    shape_size_df = pd.DataFrame({'Shape': [df.shape], 'Size': [df.size]})
    st.title('Shape and Size of dataset')
    st.write(shape_size_df)

    # (e) Data types
    data_types_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
    st.title('Data types of columns')
    st.write(data_types_df)

    # (f) Numerical features in the dataset
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_features:
        st.title('Numerical features in the dataset')
        st.write(numerical_features)

    # (g) Categorical features in the dataset
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        st.title('Categorical features in the dataset')
        st.write(categorical_features)
    else:
        st.title('No object/category data in dataset.')

    # (h) Statistical Description of Columns
    if numerical_features:
        st.title('Statistical Description of Numerical Columns')
        st.write(df.describe().T)

    # (i) Description of Categorical features
    if categorical_features:
        st.title('Description of Categorical Features')
        st.write(df.describe(include=['object', 'category']))

    # (j) Unique class count of Categorical features
    if categorical_features:
        unique_counts_df = pd.DataFrame(df[categorical_features].nunique(), columns=['Unique Count'])
        st.title('Unique class count of Categorical features')
        st.write(unique_counts_df)

    # (k) Missing values in data
    missing_values_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
    missing_values_df = missing_values_df[missing_values_df['Missing Values'] > 0]
    if not missing_values_df.empty:
        st.title('Missing values in data')
        st.write(missing_values_df)
    else:
        st.title('No missing values found.')
        st.write(df.isnull().sum())

    # (l) Unique Value Counts
    unique_values = {}
    for col in df.columns:
        unique_values[col] = df[col].nunique()

    unique_value_counts = pd.DataFrame(unique_values, index=['unique value count']).transpose()

    st.title('Unique Value Counts')
    st.write(unique_value_counts)

def html_report(df, file_path):
    # Generate profile report
    report = ProfileReport(df, title="Report", html={'style': {'full_width': True}}, explorative=True, missing_diagrams={'bar': True})
    
    # Save the report in HTML file
    report.to_file(file_path)
    return report

def plot_binary_pie(df, variabel_biner):
    fig, ax = plt.subplots(figsize=(15, 10))
    for var in variabel_biner:
        ax = plt.subplot(4, 4, variabel_biner.index(var) + 1)
        df[var].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen'], ax=ax)
        plt.title(var)
        plt.ylabel('')

    st.pyplot(fig)

def shapiro_test(data, alpha=0.05):
    stat, p = shapiro(data)
    
    if p > alpha:
        return "Sample looks Gaussian (fail to reject H0)"
    else:
        return "Sample does not look Gaussian (reject H0)"

def plot_boxplots(df, cols):
    fig, ax = plt.subplots(figsize=(12, 10))
    for i, col in enumerate(cols):
        ax = plt.subplot(4, 2, i+1)
        sns.boxplot(x=df[col], palette='Set2')
        plt.title(col)
    plt.tight_layout()
    st.pyplot(fig)

def plot_histograms(df):
    cols_with_hist = [col for col in df.columns if len(df[col].unique()) > 1]  # Filter kolom dengan lebih dari 1 nilai unik

    num_cols = len(cols_with_hist)
    num_rows = int(np.ceil(num_cols / 3))  # Hitung jumlah baris
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, num_rows*5))  # Membuat subplot dengan 3 kolom
    col_index = 0

    for i in range(num_rows):
        for j in range(3):
            if col_index < num_cols:
                col = cols_with_hist[col_index]
                axes[i, j].hist(df[col], bins='auto', color='skyblue', alpha=0.7)
                axes[i, j].set_xlabel(col)
                axes[i, j].set_ylabel('Frequency')
                axes[i, j].set_title(f'Histogram of {col}')
                col_index += 1

    plt.tight_layout()
    st.pyplot(fig)
def apply_mappings(df):
    age_mapping = {
        1: '18 to 24',
        2: '25 to 29',
        3: '30 to 34',
        4: '35 to 39',
        5: '40 to 44',
        6: '45 to 49',
        7: '50 to 54',
        8: '55 to 59',
        9: '60 to 64',
        10: '65 to 69',
        11: '70 to 74',
        12: '75 to 79',
        13: '80 or older'
    }

    diabetes_mapping = {
        0: 'Non Diabetes',
        1: 'Diabetes'
    }

    high_bp_mapping = {
        0: 'No High Blood Pressure',
        1: 'High Blood Pressure'
    }

    high_chol_mapping = {
        0: 'No High Cholesterol',
        1: 'High Cholesterol'
    }

    chol_check_mapping = {
        0: 'No Cholesterol Check in 5 Years',
        1: 'Cholesterol Check in 5 Years'
    }

    smoker_mapping = {
        0: 'Non Smoker',
        1: 'Smoker'
    }

    stroke_mapping = {
        0: 'Not Stroke',
        1: 'Stroke'
    }

    heart_disease_mapping = {
        0: 'No Heart Disease or Attack',
        1: 'Heart Disease or Attack'
    }

    phys_activity_mapping = {
        0: 'Have not physical activity or exercise',
        1: 'Do physical activity or exercise'
    }

    fruits_mapping = {
        0: 'Not Consume Fruit',
        1: 'Consume Fruit'
    }

    veggies_mapping = {
        0: 'Not Consume Vegetables',
        1: 'Consume Vegetables'
    }

    alcohol_consump_mapping = {
        0: 'Not Heavy drinkers',
        1: 'Heavy drinkers'
    }

    healthcare_mapping = {
        0: 'Not have health care coverage',
        1: 'Have health care coverage'
    }

    doc_cost_mapping = {
        0: 'Not have cost barrier to see a doctor',
        1: 'Have cost barrier to see a doctor'
    }

    general_health_mapping = {
        1: 'Poor',
        2: 'Fair',
        3: 'Good',
        4: 'Very Good',
        5: 'Excellent'
    }

    walk_diff_mapping = {
        0: 'Not have difficulty walking',
        1: 'Have difficulty walking'
    }

    sex_mapping = {
        0: 'Female',
        1: 'Male'
    }

    education_mapping = {
        1: 'Never Attended School',
        2: 'Elementary',
        3: 'Junior High School',
        4: 'Senior High School',
        5: 'Some college or technical school',
        6: 'College graduate'
    }

    income_mapping = {
        1: 'Less than $10,000',
        2: '$10,000 to $14,999',
        3: '$15,000 to $19,999',
        4: '$20,000 to $24,999',
        5: '$25,000 to $34,999',
        6: '$35,000 to $49,999',
        7: '$50,000 to $74,999',
        8: '$75,000 or More',
    }

    mappings = {
        'Age': age_mapping,
        'Diabetes_binary': diabetes_mapping,
        'HighBP': high_bp_mapping,
        'HighChol': high_chol_mapping,
        'CholCheck': chol_check_mapping,
        'Smoker': smoker_mapping,
        'Stroke': stroke_mapping,
        'HeartDiseaseorAttack': heart_disease_mapping,
        'PhysActivity': phys_activity_mapping,
        'Fruits': fruits_mapping,
        'Veggies': veggies_mapping,
        'HvyAlcoholConsump': alcohol_consump_mapping,
        'AnyHealthcare': healthcare_mapping,
        'NoDocbcCost': doc_cost_mapping,
        'GenHlth': general_health_mapping,
        'DiffWalk': walk_diff_mapping,
        'Sex': sex_mapping,
        'Education': education_mapping,
        'Income': income_mapping
    }

    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    return df

def plot_categorical_distribution(df):
    # Gradient color palette for 'Age', 'MentHlth', 'PhysHlth', and 'Income'
    gradient_palette = sns.color_palette("viridis", n_colors=len(df['Age'].unique()))

    # Colorful palette for 'GenHlth' and 'Education'
    colorful_palette = sns.color_palette("Set2")

    # Set the visualization size
    plt.figure(figsize=(24, 16))

    # Loop to create a count plot for each categorical variable
    for i, var in enumerate(['Age', 'GenHlth', 'MentHlth', 'PhysHlth', 'Education', 'Income'], 1):
        plt.subplot(3, 2, i)
        if var in ['Age', 'MentHlth', 'PhysHlth']:
            sns.countplot(x=var, data=df, palette=gradient_palette, order=df[var].value_counts().index)
        else:
            sns.countplot(x=var, data=df, palette=colorful_palette, order=df[var].value_counts().index)
        plt.title(f'Distribution of {var}')
        plt.xticks(rotation=45)

    plt.tight_layout()
    st.pyplot()

def plot_bmi_distribution(df):
    # Categorizing BMI variables
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, float('inf')], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    # Set the visualization size
    plt.figure(figsize=(12, 8))

    # Calculate the amount of data for each BMI category
    bmi_category_counts = df['BMI_Category'].value_counts()

    # Setting up colors for the histogram
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue']

    # Create a histogram for BMI categories
    bars = plt.bar(bmi_category_counts.index, bmi_category_counts.values, color=colors)

    # Added number labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height}\n({height / len(df) * 100:.1f}%)',
                 ha='center', va='bottom')

    plt.xlabel('BMI Category')
    plt.ylabel('Count')
    plt.title('Distribution of BMI Categories')

    plt.xticks(rotation=45)
    st.pyplot()

def plot_bmi_diabetes_relation(df):
    plt.figure(figsize=(25, 17))
    
    sns.distplot(df['BMI'][df['Diabetes_binary'] == 0], color="g", label="No Diabetic")
    sns.distplot(df['BMI'][df['Diabetes_binary'] == 1], color="r", label="Diabetes")
    plt.title("Relation between BMI and Diabetes")
    plt.legend()
    st.pyplot()

def create_plot_pivot(df, x_column):
    df_plot = df.groupby([x_column, 'Diabetes_binary']).size().reset_index().pivot(columns='Diabetes_binary', index=x_column, values=0)
    return df_plot

def plot_stacked_bar(df, cols):
    num_rows = len(cols) // 4 + (len(cols) % 4 > 0)
    fig, ax = plt.subplots(num_rows, 4, figsize=(30, 5 * num_rows))
    axe = ax.ravel()

    for i, col in enumerate(cols):
        df_plot = create_plot_pivot(df, col)
        df_plot.plot(kind='bar', stacked=True, ax=axe[i])
        axe[i].set_xlabel(col)
        axe[i].legend(["Non-Diabetic", "Diabetic"])

    plt.tight_layout()
    st.pyplot(fig)

def plot_diabetes_frequency_by_age(df):
    crosstab_df = pd.crosstab(df['Age'], df['Diabetes_binary'])
    crosstab_df.plot(kind="bar", figsize=(20, 6))
    plt.title('Diabetes Disease Frequency for Ages')
    plt.xlabel('Age')
    plt.xticks(rotation=0)
    plt.ylabel('Frequency')
    st.pyplot()

def plot_education_diabetes_relation(df):
    plt.figure(figsize=(10, 6))
    sns.distplot(df['Education'][df['Diabetes_binary'] == 0], color="y", label="No Diabetic")
    sns.distplot(df['Education'][df['Diabetes_binary'] == 1], color="m", label="Diabetic")
    plt.title("Relation between Education and Diabetes")
    plt.legend()
    st.pyplot()

def plot_income_diabetes_relation(df):
    plt.figure(figsize=(20, 10))
    sns.displot(data=df, x="Income", hue="Diabetes_binary", kind="kde")
    plt.title("Relation between Income and Diabetes")
    st.pyplot()

def plot_diabetes_frequency_by_genhlth(df):
    crosstab_df = pd.crosstab(df['GenHlth'], df['Diabetes_binary'])
    crosstab_df.plot(kind="bar", figsize=(30, 12), color=['Purple', 'pink'])
    plt.title('Diabetes Disease Frequency for General Health')
    plt.xlabel('GenHlth')
    plt.xticks(rotation=0)
    plt.ylabel('Frequency')
    st.pyplot()

def plot_correlation_heatmap(df):
    colors = [(0, 'darkblue'), (0.25, 'navy'), (0.5, 'white'), (0.75, 'darkred'), (1, 'black')]
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=True, cmap=cmap, vmin=-1, vmax=1)
    plt.title("Correlation of Features")
    st.pyplot()

def plot_correlation_with_target(df):
    # Compute correlation matrix
    correlation_matrix = df.drop('Diabetes_binary', axis=1).corrwith(df['Diabetes_binary'])

    # Sort correlation values in descending order
    sorted_correlation = correlation_matrix.sort_values(ascending=False)

    # Plot correlation values as a bar chart
    plt.figure(figsize=(20, 8))
    sorted_correlation.plot(kind='bar', grid=True, title="Correlation with Diabetes_binary", color="lightblue")
    plt.xlabel("Features")
    plt.ylabel("Correlation")
    st.pyplot()

    # Print sorted correlation values
    st.write("Correlation values with Diabetes_binary (sorted):")
    st.write(sorted_correlation)

def calc_VIF(df):
    X = add_constant(df)
    vif = pd.DataFrame()
    vif['variables'] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

def perform_ANOVA(df):
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]

    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=10)
    # apply feature selection
    X_selected = fs.fit_transform(X, Y)
    return X_selected.shape

def perform_ChiSquare(df):
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]

    # apply SelectKBest class to extract top 10 best features
    BestFeatures = SelectKBest(score_func=chi2, k=10)
    fit = BestFeatures.fit(X, Y)

    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)

    # concatenating two dataframes for better visualization
    f_Scores = pd.concat([df_columns, df_scores], axis=1)  # feature scores
    f_Scores.columns = ['Feature', 'Score']

    f_Scores_sorted = f_Scores.sort_values(by='Score', ascending=False)
    return f_Scores_sorted
