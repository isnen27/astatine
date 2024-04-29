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

# from pandas_profiling import ProfileReport
from pandas.plotting import parallel_coordinates
from IPython.display import display, Markdown

# for model preparation
import sklearn
from sklearn.model_selection import train_test_split
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

#Evaluation
from sklearn.metrics import confusion_matrix,RocCurveDisplay, classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#Save model
import joblib
import warnings
import pickle
warnings.filterwarnings('ignore')

def svm_model(X_train, Y_train, X_test, Y_test):
    # Define and train the model
    svm = SVC(kernel='rbf', C=1.0)
    svm.fit(X_train, Y_train)

    # Predictions
    y_pred_svm = svm.predict(X_test)
    
    # Calculate scores
    train_score = svm.score(X_train, Y_train)
    test_score = svm.score(X_test, Y_test)

    # Calculate MSE & RMSE
    mse_svm = mean_squared_error(Y_test, y_pred_svm)
    rmse_svm = np.sqrt(mse_svm)

    # Generate classification report
    report = classification_report(Y_test, y_pred_svm, output_dict=True)

    # Calculate confusion matrix
    cm_svm = confusion_matrix(Y_test, y_pred_svm)

    # Display scores
    st.write('**SVM Evaluation**')
    st.write('Training set score: {:.4f}'.format(train_score))
    st.write('Test set score: {:.4f}'.format(test_score))
    st.write('Mean Squared Error: {:.4f}'.format(mse_svm))
    st.write('Root Mean Squared Error: {:.4f}'.format(rmse_svm))
    st.write('\n')

    # Display classification report
    st.write('Classification Report:')
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    # Plot confusion matrix
    st.write('Confusion Matrix:')
    plot_confusion_matrix_svm(cm_svm)

    # Save the model
    joblib.dump(svm, 'svm_model.pkl')
    st.write("SVM model saved successfully!")

def plot_confusion_matrix_svm(cm, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if classes is None:
        classes = np.arange(cm.shape[0])
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        st.write("Normalized confusion matrix")
    else:
        st.write('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    # Label sumbu x dan y
    if classes is None:
        classes = np.unique(np.concatenate((Y_train, Y_test), axis=None))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot()


def xgboost_model(X_train, Y_train, X_test, Y_test):
    # Define the model
    xg = XGBClassifier(eval_metric='error', learning_rate=0.1)
    xg.fit(X_train, Y_train)

    # Predictions
    y_pred_xg = xg.predict(X_test)
    
    # Calculate scores
    train_score = xg.score(X_train, Y_train)
    test_score = xg.score(X_test, Y_test)

    # Calculate MSE & RMSE
    mse_xg = mean_squared_error(Y_test, y_pred_xg)
    rmse_xg = np.sqrt(mse_xg)

    # Generate classification report
    report = classification_report(Y_test, y_pred_xg, output_dict=True)
    
    # Calculate confusion matrix
    cm_xg = confusion_matrix(Y_test, y_pred_xg)
    
    # Display scores
    st.write('**XGBoost Evaluation**')
    st.write('Training set score: {:.4f}'.format(train_score))
    st.write('Test set score: {:.4f}'.format(test_score))
    st.write('Mean Squared Error: {:.4f}'.format(mse_xg))
    st.write('Root Mean Squared Error: {:.4f}'.format(rmse_xg))
    st.write('\n')

    # Display classification report as a table
    st.write('Classification Report:')
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    # Plot confusion matrix
    plot_confusion_matrix_xg(cm_xg, classes=np.unique(Y_test))
    
    # Save the model
    joblib.dump(xg, 'xgboost_model.pkl')
    st.write("XGBoost model saved successfully!")

def plot_confusion_matrix_xg(cm, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        st.write("Normalized confusion matrix")
    else:
        st.write('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot()

def random_forest_model(X_train, Y_train, X_test, Y_test):
    # Define the model
    rf = RandomForestClassifier(max_depth=12, n_estimators=10, random_state=42)

    # Fitting the model on the train data
    rf.fit(X_train, Y_train)

    # Make predictions on the test set
    y_pred_rf = rf.predict(X_test)

    # Calculate scores
    train_score = rf.score(X_train, Y_train)
    test_score = rf.score(X_test, Y_test)

    # Check MSE & RMSE 
    mse_rf = mean_squared_error(Y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(Y_test, y_pred_rf))

    # Generate classification report
    report = classification_report(Y_test, y_pred_rf, output_dict=True)

    # Calculate confusion matrix
    cm_rf = confusion_matrix(Y_test, y_pred_rf)
    
    # Display scores
    st.write('**Random Forest Evaluation**')
    st.write('Training set score: {:.4f}'.format(train_score))
    st.write('Test set score: {:.4f}'.format(test_score))
    st.write('Mean Squared Error: {:.4f}'.format(mse_rf))
    st.write('Root Mean Squared Error: {:.4f}'.format(rmse_rf))
    st.write('\n')

    # Display classification report as a table
    st.write('Classification Report:')
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    # Plot confusion matrix
    plot_confusion_matrix_rf(cm_rf)

    # Save the model
    joblib.dump(rf, 'random_forest_model.pkl')
    st.write("Random Forest model saved successfully!")

def plot_confusion_matrix_rf(cm, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if classes is None:
        classes = np.arange(cm.shape[0])
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        st.write("Normalized confusion matrix")
    else:
        st.write('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot()

def naive_bayes_model(X_train, Y_train, X_test, Y_test):
    # Define the model
    gnb = GaussianNB()

    # Train the model
    gnb.fit(X_train, Y_train)

    # Make predictions
    y_pred_gnb = gnb.predict(X_test)

    # Calculate scores
    train_score = gnb.score(X_train, Y_train)
    test_score = gnb.score(X_test, Y_test)

    # Check MSE & RMSE
    mse_gnb = mean_squared_error(Y_test, y_pred_gnb)
    rmse_gnb = np.sqrt(mse_gnb)

    # Generate classification report
    report = classification_report(Y_test, y_pred_gnb, output_dict=True)

    # Calculate confusion matrix
    cm_gnb = confusion_matrix(Y_test, y_pred_gnb)
    
    # Display scores
    st.write('**Naive Bayes Evaluation**')
    st.write('Training set score: {:.4f}'.format(train_score))
    st.write('Test set score: {:.4f}'.format(test_score))
    st.write('Mean Squared Error: {:.4f}'.format(mse_gnb))
    st.write('Root Mean Squared Error: {:.4f}'.format(rmse_gnb))
    st.write('\n')

    # Display classification report as a table
    st.write('Classification Report:')
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    # Plot confusion matrix
    plot_confusion_matrix_gnb(cm_gnb)

    # Save the model
    joblib.dump(gnb, 'naive_bayes_model.pkl')
    st.write("Naive Bayes model saved successfully!")

def plot_confusion_matrix_gnb(cm, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if classes is None:
        classes = np.arange(cm.shape[0])
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        st.write("Normalized confusion matrix")
    else:
        st.write('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot()

def ann_model(X_train, Y_train, X_test, Y_test):
    # Define the model
    model_ann = Sequential()
    model_ann.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))  # Input layer
    model_ann.add(Dense(64, activation='relu'))  # Hidden layer
    model_ann.add(Dense(1, activation='sigmoid'))  # Output layer

    # Compile the model
    model_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model_ann.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test), verbose=0)

    # Evaluate the model
    train_score_ann = model_ann.evaluate(X_train, Y_train, verbose=0)
    test_score_ann = model_ann.evaluate(X_test, Y_test, verbose=0)

    # Display scores
    st.write('**ANN Evaluation**')
    st.write('Training set accuracy: {:.4f}'.format(train_score_ann[1]))
    st.write('Test set accuracy: {:.4f}'.format(test_score_ann[1]))
    st.write('\n')

    # Predictions
    y_pred_prob_ann = model_ann.predict(X_test)
    y_pred_ann = (y_pred_prob_ann > 0.5).astype(int)

    # Check MSE & RMSE
    mse_ann = mean_squared_error(Y_test, y_pred_ann)
    rmse_ann = np.sqrt(mse_ann)

    # Generate classification report
    report = classification_report(Y_test, y_pred_ann, output_dict=True)

    # Display classification report as a table
    st.write('Classification Report:')
    df_report = pd.DataFrame(report).transpose()
    st.write(df_report)

    # Calculate confusion matrix
    cm_ann = confusion_matrix(Y_test, y_pred_ann)
    
    # Plot confusion matrix
    plot_confusion_matrix_ann(cm_ann)

    # Save the model
    model_save_path = "ann_model.keras"
    model_ann.save(model_save_path)
    st.write("ANN model saved successfully!")

def plot_confusion_matrix_ann(cm, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if classes is None:
        classes = np.arange(cm.shape[0])
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        st.write("Normalized confusion matrix")
    else:
        st.write('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    st.pyplot()

def evaluate_models(svm, xg, rf, gnb, train_score_ann, test_score_ann, mse_svm, mse_xg, mse_rf, mse_gnb, mse_ann, rmse_svm, rmse_xg, rmse_rf, rmse_gnb, rmse_ann):
    # Data for all models
    models_data = {
        'Model': ['SVM', 'XGBoost', 'Random Forest', 'Naive Bayes', 'ANN'],
        'Training set accuracy': [svm.score(X_train, Y_train), xg.score(X_train, Y_train),rf.score(X_train, Y_train), gnb.score(X_train, Y_train),
                                  train_score_ann[1]],
        'Test set accuracy': [svm.score(X_test, Y_test), xg.score(X_test, Y_test),
                              rf.score(X_test, Y_test), gnb.score(X_test, Y_test),
                              test_score_ann[1]],
        'MSE': [mse_svm, mse_xg, mse_rf, mse_gnb, mse_ann],
        'RMSE': [rmse_svm, rmse_xg, rmse_rf, rmse_gnb, rmse_ann]
    }

    # Create DataFrame
    models_df = pd.DataFrame(models_data)

    # Sorting models based on Test set accuracy
    sorted_models = models_df.sort_values(by='Test set accuracy', ascending=False)

    return sorted_models

# for astatine.py:
# sorted_models = evaluate_models(svm, xg, rf, gnb, train_score_ann, test_score_ann, mse_svm, mse_xg, mse_rf, mse_gnb, mse_ann, rmse_svm, rmse_xg, rmse_rf, rmse_gnb, rmse_ann)
# print(sorted_models)

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_auc(X_test, y_pred_xg, y_pred_rf, y_pred_gnb, y_pred_ann):
    # Dictionary to store AUC scores for each model
    models_roc_auc = {}

    # XGBoost
    fpr_xg, tpr_xg, _ = roc_curve(Y_test, y_pred_xg)
    roc_auc_xg = auc(fpr_xg, tpr_xg)
    models_roc_auc['XGBoost'] = roc_auc_xg
    plt.plot(fpr_xg, tpr_xg, label='XGBoost (AUC = %0.2f)' % roc_auc_xg)

    # Random Forest
    fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_pred_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    models_roc_auc['Random Forest'] = roc_auc_rf
    plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = %0.2f)' % roc_auc_rf)

    # Naive Bayes
    fpr_gnb, tpr_gnb, _ = roc_curve(Y_test, y_pred_gnb)
    roc_auc_gnb = auc(fpr_gnb, tpr_gnb)
    models_roc_auc['Naive Bayes'] = roc_auc_gnb
    plt.plot(fpr_gnb, tpr_gnb, label='Naive Bayes (AUC = %0.2f)' % roc_auc_gnb)

    # ANN
    fpr_ann, tpr_ann, _ = roc_curve(Y_test, y_pred_ann)
    roc_auc_ann = auc(fpr_ann, tpr_ann)
    models_roc_auc['ANN'] = roc_auc_ann
    plt.plot(fpr_ann, tpr_ann, label='ANN (AUC = %0.2f)' % roc_auc_ann)

    # Plotting ROC curve
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Display AUC for each model
    st.write("Area under the Curve (AUC) for each model:")
    for model, auc_score in models_roc_auc.items():
        st.write(f"{model}: {auc_score}")

# for astatine.py:
# models = {'SVM': svm, 'XGBoost': xg, 'Random Forest': rf, 'Naive Bayes': gnb, 'ANN': model_ann}
# y_preds = [y_pred_svm, y_pred_xg, y_pred_rf, y_pred_gnb, y_pred_ann]
# plot_roc_curve(models, Y_test, y_preds)
