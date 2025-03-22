from diabetesModelCommon import clean_data, load_data, \
    feature_selection, select_best_features, populateColumnImportanceScore, \
    randomForestModelGridSearch, logModelScore, dumpModelBestColumn, dumpColumnMeanValues, \
    remove_outliers_from_each_column
from sklearn.model_selection import train_test_split
import os

parent_directory = os.path.dirname(os.getcwd())

print("Model training started.......")

train_dataset = os.path.join(os.getcwd(), "data.csv")

# Load the data and perform data cleaning
data = load_data(train_dataset)

# Clean data
data = clean_data(data)

# select feature features for converted data
converted_data_X, converted_data_y = feature_selection(data)

clean_converted_data_X = remove_outliers_from_each_column(converted_data_X)

# select feature features
X, y = feature_selection(data)

# Select best matrix
X, selected_columns = select_best_features(X, y)

# Save the selected columns to a file using joblib
dumpModelBestColumn(selected_columns)

# Scale and split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = randomForestModelGridSearch(X_train, y_train)

# Predict model test values
y_pred = model.predict(X_test)

# Predict the conversion probabilities for the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Log feature importance score
populateColumnImportanceScore(model.best_estimator_, selected_columns)

logModelScore(y_test, y_pred, y_pred_proba)

print("Model training completed.......")
