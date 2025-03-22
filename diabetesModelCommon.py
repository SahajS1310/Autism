import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import joblib
import numpy as np
from scipy import stats
import os

dir = os.getcwd()

feature_column = ['Age of mother while conceiving', 'Age of Father at that time',
                  'medication of mother during pregnancy, If any', 'Mode of delivery',
                  'Health of the mother while Conceiving and during pregnancy',
                  'Any neurodevelopmental condition present or specially abled member in the family/ family history.']


def load_data(filepath):
    """Load data from a CSV file into a Pandas DataFrame"""
    data = pd.read_csv(filepath, low_memory=False)
    return data


def clean_data(data):
    """Clean data by removing duplicates and filling in missing values"""
    data.drop_duplicates(inplace=True)
    data.fillna(0, inplace=True)
    return data


def remove_outliers_ZScore(data_X, threshold=3):
    z_scores = np.abs(stats.zscore(data_X))
    rows_to_remove = np.any(z_scores > threshold, axis=1)
    cleaned_data_X = data_X[~rows_to_remove]
    return cleaned_data_X


def remove_outliers_from_each_column(data_X, threshold=3):
    # Create a copy of the input data to avoid modifying the original DataFrame
    cleaned_data_X = data_X.copy()

    # Remove outliers for each column of 'data_X'
    for col in data_X.columns:
        cleaned_data_X_col = remove_outliers_ZScore(data_X[[col]], threshold=threshold)
        cleaned_data_X[col] = cleaned_data_X_col

    return cleaned_data_X


def feature_engineering(data):
    """Perform feature engineering on the DataFrame"""
    if 'total_webinar_count' not in data.columns:
        data['total_webinar_count'] = data['webinar_daily_count'] + data['webinar_weekly_count']
    return data


def select_best_features(X, y):
    # Perform feature selection using SelectKBest and chi2
    selector = SelectKBest(chi2, k=6)
    selector.fit(X, y)
    X_new = selector.transform(X)
    selected_columns = X.columns[selector.get_support()]
    print("Best feature matrix : {}".format(selected_columns))
    return X_new, selected_columns


def handle_categorical_variable(X):
    X = pd.get_dummies(X, columns=['l_score', 'SQL'])
    if 'l_score_0' not in X.columns:
        X.insert(loc=6,
                 column='l_score_0',
                 value=0)
    if 'SQL_0' not in X.columns:
        X.insert(loc=10,
                 column='SQL_0',
                 value=0)
    return X


def remove_outliers(df, column):
    """Remove outliers from a column of a DataFrame using the IQR method"""
    Q1 = np.percentile(df[column], 25)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def feature_selection(data):
    """Perform feature selection on the DataFrame"""
    X = data[feature_column]
    y = data[
        'How much attention does the child pays?'] if 'How much attention does the child pays?' in data.columns else None

    return X, y


def scale_data(X):
    """Scale and normalize the data using StandardScaler"""
    scaler = StandardScaler()
    # MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def populateColumnImportanceScore(model, selected_columns):
    # Get the feature importances
    feature_importances = model.feature_importances_
    # Create a dictionary of feature importances
    importance_dict = dict(zip(selected_columns, feature_importances))
    # Sort importance score
    sorted_importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    # Log the importance score of each column
    for col, importance in sorted_importance_dict.items():
        print("Importance of column {}: {}".format(col, importance))

    return feature_importances


def randomForestModelGridSearch(X_train, y_train):
    print("Selected Random forest model")
    # Define the hyperparameters for tuning
    """
        n_estimators: Number of decision trees in the forest.
        max_depth: Maximum depth of each decision tree.
        min_samples_split: Minimum number of samples required to split a node.
        min_samples_leaf: Minimum number of samples required to be at a leaf node.
    """
    param_grid = {'n_estimators': [10, 50, 100],
                  'max_depth': [None, 10, 20],
                  'min_samples_split': [2, 4, 6, 8],
                  'min_samples_leaf': [1, 2, 3]}

    # Create a random forest classifier and perform hyperparameter tuning
    """"
        class_weight='balanced': This balances the weight of the classes by adjusting the weights inversely proportional 
        to the class frequencies in the input data.

        random_state=42: This sets the random seed to a fixed value (42) for reproducibility purposes.
    """
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)

    """"
        GridSearchCV is a method in scikit-learn that performs a search over specified hyperparameters 
        for an estimator using cross-validation. 

        rf is a RandomForestClassifier instance with balanced class weights and a random seed of 42.

        param_grid is a dictionary that contains the hyperparameters to be tuned for the random forest classifier.

        cv is set to 5, which means that the method performs a 5-fold cross-validation.

        n_jobs is set to -1, which means that the method uses all available CPU cores to parallelize the grid search.

        scoring is set to 'roc_auc', which means that the method uses the ROC AUC score to evaluate the performance of the model.
    """
    grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    # Print the best hyperParameters and the corresponding AUC score
    print('Best hyperParameters: {}'.format(grid_search.best_params_))
    print('Best AUC score: {}'.format(grid_search.best_score_))

    # Assuming you have trained and tuned your model as 'grid_search'
    joblib.dump(grid_search.best_estimator_, "model.joblib")

    return grid_search

def logModelScore(y_test, y_pred, y_pred_proba):
    # Print the accuracy score, confusion matrix, classification report, and AUC score
    print('Accuracy: {}'.format(accuracy_score(y_test, y_pred) * 100))

    """
        A confusion matrix is a table used to evaluate the performance of a classification model by comparing the predicted 
        and actual classes. 
        It consists of four cells: true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).
    """
    print('Confusion matrix:\n {}'.format(confusion_matrix(y_test, y_pred)))

    """
        The classification report is a text report that summarizes the main classification metrics 
        (precision, recall, F1-score, and support) for each class in the target variable.

        Precision is the ratio of true positives to the sum of true positives and false positives. 
        It measures the proportion of predicted positives that are actually positive.

        Recall is the ratio of true positives to the sum of true positives and false negatives. 
        It measures the proportion of actual positives that are correctly predicted as positive.

        F1-score is the harmonic mean of precision and recall. It combines precision and recall into a single 
        metric that balances both measures.
    """
    print('Classification report:\n {}'.format(classification_report(y_test, y_pred)))

    """
        This code computes the AUC score (Area Under the ROC Curve) which is a performance metric used for 
        binary classification problems. The AUC score ranges from 0 to 1, with higher values indicating better 
        classification performance.
    """
    print('AUC score: {}'.format(roc_auc_score(y_test, y_pred_proba)))


def dumpModelBestColumn(selected_columns):
    # Save the selected columns to a file using joblib
    joblib.dump(selected_columns, "selected_columns.joblib")
