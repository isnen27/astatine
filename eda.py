# for basic operations
import numpy as np
import pandas as pd

from ydata_profiling import ProfileReport
from pandas.plotting import parallel_coordinates
from IPython.display import display, Markdown

def describe_detail(df):
    # Display function with Markdown for titles
    def display_markdown(title):
        display(Markdown(f"**{title}**"))

    # (a) First five data points
    display_markdown('First five data points')
    display(df.head())
    print('\n')

    # (b) Random five data points
    display_markdown('Random five data points')
    display(df.sample(5))
    print('\n')

    # (c) Last five data points
    display_markdown('Last five data points')
    display(df.tail())
    print('\n')

    # (d) Shape and Size of data set
    shape_size_df = pd.DataFrame({'Shape': [df.shape], 'Size': [df.size]})
    display_markdown('Shape and Size of dataset')
    display(shape_size_df)
    print('\n')

    # (e) Data types
    data_types_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
    display_markdown('Data types of columns')
    display(data_types_df)
    print('\n')

    # (f) Numerical features in the dataset
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_features:
        display_markdown('Numerical features in the dataset')
        display(numerical_features)
        print('\n')

    # (g) Categorical features in the dataset
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        display_markdown('Categorical features in the dataset')
        display(categorical_features)
        print('\n')
    else:
       display_markdown('**No object/category data in dataset.**')
       print('\n')

    # (h) Statistical Description of Columns
    if numerical_features:
        display_markdown('Statistical Description of Numerical Columns')
        display(df.describe().T)
        print('\n')

    # (i) Description of Categorical features
    if categorical_features:
        display_markdown('Description of Categorical Features')
        display(df.describe(include=['object', 'category']))
        print('\n')

    # (j) Unique class count of Categorical features
    if categorical_features:
        unique_counts_df = pd.DataFrame(df[categorical_features].nunique(), columns=['Unique Count'])
        display_markdown('Unique class count of Categorical features')
        display(unique_counts_df)
        print('\n')

    # (k) Missing values in data
    missing_values_df = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
    missing_values_df = missing_values_df[missing_values_df['Missing Values'] > 0]
    if not missing_values_df.empty:
        display_markdown('Missing values in data')
        display(missing_values_df)
    else:
        display_markdown('**No missing values found.**')
        display(df.isnull().sum())
        print('\n')

    # (l) Unique Value Counts
    unique_values = {}
    for col in df.columns:
        unique_values[col] = df[col].nunique()

    unique_value_counts = pd.DataFrame(unique_values, index=['unique value count']).transpose()

    display_markdown('**Unique Value Counts**')
    display(unique_value_counts)
  
def html_report(df, file_path):
    # Generate profile report
    report = ProfileReport(df, title="Report", html={'style': {'full_width': True}}, explorative=True, missing_diagrams={'bar': True})
    
    # Save the report in HTML file
    report.to_file(file_path)
