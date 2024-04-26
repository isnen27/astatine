# for basic operations
import numpy as np
import pandas as pd

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
    # define the model
    svm = SVC(kernel='rbf', C=1.0)
    
    # train the model
    svm.fit(X_train, Y_train)
    
    y_pred_svm = svm.predict(X_test)
    
    print('SVM Evaluation')
    print('Training set score: {:.4f}'.format(svm.score(X_train, Y_train)))
    print('Test set score: {:.4f}'.format(svm.score(X_test, Y_test)))
    
    # check MSE & RMSE 
    mse_svm = mean_squared_error(Y_test, y_pred_svm)
    print('Mean Squared Error : '+ str(mse_svm))
    rmse_svm = math.sqrt(mean_squared_error(Y_test, y_pred_svm))
    print('Root Mean Squared Error : '+ str(rmse_svm))
    print('\n')
    
    matrix = classification_report(Y_test, y_pred_svm)
    print(matrix)
    
    # calculating and plotting the confusion matrix
    cm_svm = confusion_matrix(Y_test, y_pred_svm)
    plot_confusion_matrix_svm(cm_svm)
    plt.show()

def plot_confusion_matrix_svm(cm, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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

def xgboost_model(X_train, Y_train, X_test, Y_test):
    xg = XGBClassifier(eval_metric='error', learning_rate=0.1)
    xg.fit(X_train, Y_train)

    y_pred_xg = xg.predict(X_test)
    print('XGBoost Evaluation')
    print('Training set score: {:.4f}'.format(xg.score(X_train, Y_train)))
    print('Test set score: {:.4f}'.format(xg.score(X_test, Y_test)))

    # check MSE & RMSE 
    mse_xg = mean_squared_error(Y_test, y_pred_xg)
    print('Mean Squared Error : '+ str(mse_xg))
    rmse_xg = math.sqrt(mean_squared_error(Y_test, y_pred_xg))
    print('Root Mean Squared Error : '+ str(rmse_xg))
    print('\n')

    matrix = classification_report(Y_test, y_pred_xg)
    print(matrix)

    # calculating and plotting the confusion matrix
    cm_xg = confusion_matrix(Y_test, y_pred_xg)
    plot_confusion_matrix_xg(cm_xg)
    plt.show()

def plot_confusion_matrix_xg(cm, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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

def random_forest_model(X_train, Y_train, X_test, Y_test):
    rf = RandomForestClassifier(max_depth=12, n_estimators=10, random_state=42)

    # fitting the model on the train data
    rf.fit(X_train, Y_train)

    # make predictions on test set
    y_pred_rf = rf.predict(X_test)

    print('Random Forest Evaluation')
    print('Training set score: {:.4f}'.format(rf.score(X_train, Y_train)))
    print('Test set score: {:.4f}'.format(rf.score(X_test, Y_test)))

    # check MSE & RMSE 
    mse_rf = mean_squared_error(Y_test, y_pred_rf)
    print('Mean Squared Error : '+ str(mse_rf))
    rmse_rf = math.sqrt(mean_squared_error(Y_test, y_pred_rf))
    print('Root Mean Squared Error : '+ str(rmse_rf))
    print('\n')

    matrix = classification_report(Y_test, y_pred_rf)
    print(matrix)

    # calculating and plotting the confusion matrix
    cm_rf = confusion_matrix(Y_test, y_pred_rf)
    plot_confusion_matrix_rf(cm_rf)
    plt.show()

def plot_confusion_matrix_rf(cm, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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

def naive_bayes_model(X_train, Y_train, X_test, Y_test):
    # define the model
    gnb = GaussianNB()

    # train the model
    gnb.fit(X_train, Y_train)

    y_pred_gnb = gnb.predict(X_test)

    print('Naive Bayes Evaluation')
    print('Training set score: {:.4f}'.format(gnb.score(X_train, Y_train)))
    print('Test set score: {:.4f}'.format(gnb.score(X_test, Y_test)))

    # check MSE & RMSE
    mse_gnb = mean_squared_error(Y_test, y_pred_gnb)
    print('Mean Squared Error : ' + str(mse_gnb))
    rmse_gnb = math.sqrt(mean_squared_error(Y_test, y_pred_gnb))
    print('Root Mean Squared Error : ' + str(rmse_gnb))
    print('\n')

    matrix = classification_report(Y_test, y_pred_gnb)
    print(matrix)

    # calculating and plotting the confusion matrix
    cm_gnb = confusion_matrix(Y_test, y_pred_gnb)
    plot_confusion_matrix_gnb(cm_gnb)
    plt.show()

def plot_confusion_matrix_gnb(cm, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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

    print('ANN Evaluation')
    print('Training set accuracy: {:.4f}'.format(train_score_ann[1]))
    print('Test set accuracy: {:.4f}'.format(test_score_ann[1]))

    # Predictions
    y_pred_prob_ann = model_ann.predict(X_test)
    y_pred_ann = (y_pred_prob_ann > 0.5).astype(int)

    # Check MSE & RMSE
    mse_ann = mean_squared_error(Y_test, y_pred_ann)
    print('Mean Squared Error: {:.4f}'.format(mse_ann))
    rmse_ann = math.sqrt(mse_ann)
    print('Root Mean Squared Error: {:.4f}'.format(rmse_ann))
    print('\n')

    # Classification report
    matrix = classification_report(Y_test, y_pred_ann)
    print(matrix)

    # Calculating and plotting the confusion matrix
    cm_ann = confusion_matrix(Y_test, y_pred_ann)
    plot_confusion_matrix_ann(cm_ann, show_absolute=True, show_normed=True, colorbar=True)
    plt.show()

def plot_confusion_matrix_ann(cm, classes=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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

def evaluate_models(svm, xg, rf, gnb, train_score_ann, test_score_ann, mse_svm, mse_xg, mse_rf, mse_gnb, mse_ann,
                    rmse_svm, rmse_xg, rmse_rf, rmse_gnb, rmse_ann):
    # Data for all models
    models_data = {
        'Model': ['SVM', 'XGBoost', 'Random Forest', 'Naive Bayes', 'ANN'],
        'Training set accuracy': [svm.score(X_train, Y_train), xg.score(X_train, Y_train),
                                  rf.score(X_train, Y_train), gnb.score(X_train, Y_train),
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
# sorted_models = evaluate_models(svm, xg, rf, gnb, train_score_ann, test_score_ann, mse_svm, mse_xg, mse_rf, mse_gnb, mse_ann,
#                                  rmse_svm, rmse_xg, rmse_rf, rmse_gnb, rmse_ann)
# print(sorted_models)

def plot_roc_curve(models, Y_test, y_preds):
    plt.figure(figsize=(10, 8))
    models_roc_auc = {}

    for model_name, y_pred in zip(models.keys(), y_preds):
        fpr, tpr, _ = roc_curve(Y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        models_roc_auc[model_name] = roc_auc
        plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (model_name, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Display AUC for each model
    print("Area under the Curve (AUC) for each model:")
    for model, auc_score in models_roc_auc.items():
        print(f"{model}: {auc_score}")

# for astatine.py:
# models = {'SVM': svm, 'XGBoost': xg, 'Random Forest': rf, 'Naive Bayes': gnb, 'ANN': model_ann}
# y_preds = [y_pred_svm, y_pred_xg, y_pred_rf, y_pred_gnb, y_pred_ann]
# plot_roc_curve(models, Y_test, y_preds)
