from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

#Route For Home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Temperature = int(request.form.get('Temperature')),
            Humidity = int(request.form.get('Humidity')),
            Moisture = int(request.form.get('Moisture')),
            Soil_Type = request.form.get('Soil_Type'),
            Crop_Type = request.form.get('Crop_Type'),
            Nitrogen = int(request.form.get('Nitrogen')),
            Potassium = int(request.form.get('Potassium')),
            Phosphorous = int(request.form.get('Phosphorous'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)


