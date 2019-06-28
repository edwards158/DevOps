from flask import Flask, render_template,url_for,request
from flask_material import Material

import pandas as pd 
import numpy as np 
from sklearn.externals import joblib

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preview')
def preview():
    df = pd.read_csv("data/diabetes_data.csv",usecols = [1,2,3,4,5])
    return render_template('preview.html',df_view=df)

@app.route('/analysis')
def analysis():
	if request.method == 'POST':
		bmi = request.form['bmi']
		age = request.form['age']
		pressure = request.form['pressure']
		skin_thickness = request.form['skin_thickness']
		model_choice = request.form['model_choice']

		# Clean the data by convert from unicode to float 
		sample_data = [bmi,age,pressure,skin_thickness]
		clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
		ex1 = np.array(clean_data).reshape(1,-1)

		# Reloading the Model
		if model_choice == 'logitmodel':
		    logit_model = joblib.load('data/lr_model_iris.pkl')
		    result_prediction = logit_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('data/knn_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'dtmodel':
			knn_model = joblib.load('data/dt_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)

	return render_template('index.html', bmi=bmi,
		age=age,
		pressure=pressure,
		skin_thickness=skin_thickness,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)