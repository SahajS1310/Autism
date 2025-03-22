from flask import Flask, render_template, request, jsonify
from diabetesModelTest import predict_score_and_outcome
from sql import insert_data_into_table

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_values = {
            "Age of mother while conceiving": int(request.form['age_mother']),
            "Age of Father at that time": int(request.form['age_father']),
            "medication of mother during pregnancy, If any": int(request.form['medication']),
            "Mode of delivery": int(request.form['delivery_mode']),
            "Health of the mother while Conceiving and during pregnancy": int(request.form['health']),
            "Any neurodevelopmental condition present or specially abled member in the family/ family history.": int(request.form['neurodevelopmental']),
            'Cognition and memory level of the child': int(request.form['cognition_memory']),
            'Anxiety level of child': int(request.form['anxiety_level']),
            'Speech level of child': int(request.form['speech_level']),
            'Social interaction/Communication of the child': int(request.form['social_interaction']),
            'How much attention does the child pays': int(request.form['child_pays'])
        }

        insert_data_into_table(input_values)
        outcome = predict_score_and_outcome(input_values)
        return render_template('result.html', outcome=outcome)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
