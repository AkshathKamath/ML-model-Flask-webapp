from flask import Flask,render_template,request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

## Load the model
def load_model():
    model_path = './Model/Trained-Models/RF_loan_model.joblib'
    model = joblib.load(model_path)
    return model

## Parse input form data
def parseData(reqData):
    del reqData['First_Name']
    del reqData['Last_Name']
    reqData = {k:int(v) for k,v in reqData.items()}
    data = pd.DataFrame([reqData])
    data['TotalIncome'] = data['applicant_income'] + data['co_applicant_income']
    # data['TotalIncome'] = np.log(data['TotalIncome']).copy()
    data = data[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area','TotalIncome']]
    return data

## Predict
def predictModel(model, data):
    pred = model.predict(data)
    pred = int(pred[0])
    if pred == 1:
        pred = "Congratulations! Your loan request is approved."
    if pred == 0:
        pred = "Sorry! Your loan request is rejected."
    return pred


## Views
@app.route('/') ## Root path
def home():
    return render_template('prediction.html')

@app.route('/predict',methods=['POST'])
def predict():
    model = load_model()
    if request.method == 'POST':
        # request_data = dict(request.form) 
        data = parseData(dict(request.form)) ## Convert to python dict

        # pred = model.predict(data)
        # pred = pred[0]
        pred = predictModel(model, data)
        return render_template('prediction.html',prediction = pred)
    
    
@app.errorhandler(500)
def internal_error(error):
    return "500: Something went wrong"

@app.errorhandler(404)
def not_found(error):
    return "404: Page not found",404
    

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)