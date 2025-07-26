from flask import Flask, request, render_template
import joblib
import numpy as np

print(np.__version__)

app = Flask(__name__)
model = joblib.load('cost_estimation_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            raw_material = float(request.form['raw_material'])
            labor_hours = float(request.form['labor_hours'])
            machine_hours = float(request.form['machine_hours'])
            overhead_cost = float(request.form['overhead_cost'])
            complexity = int(request.form['complexity'])

            features = [[raw_material, labor_hours, machine_hours, overhead_cost, complexity]]
            prediction = model.predict(features)[0]

        except Exception as e:
            prediction = f'Error: {str(e)}'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)