import joblib
import pandas as pd
import os
from datetime import datetime

current_date = datetime.now().date()

parent_directory = os.path.dirname(os.getcwd())

"""
    - model: Trained machine learning model
    - selected_columns: List of selected columns used during training
"""
model = joblib.load("model.joblib")
selected_columns = joblib.load("selected_columns.joblib")


def predict_score_and_outcome(input_values):
    """
    Predicts the risk score and outcome for given input values using the trained model.

    Parameters:
    - input_values: Dictionary containing input values for each selected column

    Returns:
    - risk_score: Predicted risk score
    - outcome: Predicted outcome (0 or 1)
    """

    # Create a DataFrame with the input values
    input_data = pd.DataFrame([input_values], columns=selected_columns)

    # Ensure the input data has the same columns as the training data
    input_data = input_data[selected_columns]

    # Predict the risk score and outcome
    outcome = model.predict(input_data)

    if outcome[0] == 1:
        return "Low"
    elif outcome[0] == 2:
        return "Moderately Low"
    elif outcome[0] == 3:
        return "Moderate"
    elif outcome[0] == 4:
        return "High"
    elif outcome[0] == 5:
        return "Severe"
    return outcome[0]
