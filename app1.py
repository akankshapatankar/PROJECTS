import joblib
import numpy as np
import os

from flask import Flask, jsonify, render_template,request

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("home.html")


@app.route('/predict', methods=['POST', 'GET'])
def result():
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])

    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp, outlet_establishment_year,
                   outlet_size, outlet_location_type, outlet_type]])


    sc = joblib.load(r'X:\Data_Science\Akanksha_Skillslash\Python\Bigmart_Machine_Learning_Project\models\sc.sav')

    X_std = sc.transform(X)

    model_path = r'X:\Data_Science\Akanksha_Skillslash\Python\Bigmart_Machine_Learning_Project\models\gd.sav'

    model = joblib.load(model_path)

    Y_pred = model.predict(X_std)

    return jsonify({'Prediction': float(Y_pred)})


if __name__ == "__main__":
    app.run(debug=True, port=9457)
