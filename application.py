import os
print(os.getcwd())

import pickle
from flask import Flask,request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge_model.pkl','rb')) 
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/') 
def index():
    return render_template("index.html") ## render the index.html page

@app.route('/predictdata',methods=['GET','POST']) 
def predict_datapoint():
    if request.method == "POST": 
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain")) 
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))
        
        new_data_scaled = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]) ## scale the data using standard scaler
        result = ridge_model.predict(new_data_scaled) ## predict using ridge model
        
        return render_template("home.html",results=result[0]) ## render the home.html page with the prediction result
        
    else: 
        return render_template("home.html") ## render the home.html page

if __name__=="__main__": ## Execution starts here
    app.run(host="0.0.0.0")