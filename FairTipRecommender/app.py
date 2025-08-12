from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("model/tip_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    total_bill = float(request.form["total_bill"])
    service_rating = int(request.form["service_rating"])
    time_of_day = request.form["time_of_day"]
    gender = request.form["gender"]
    group_size = int(request.form["group_size"])

    input_data = pd.DataFrame([{
        "total_bill": total_bill,
        "service_rating": service_rating,
        "time_of_day": time_of_day,
        "gender": gender,
        "group_size": group_size
    }])

    prediction = model.predict(input_data)[0]
    prediction = round(prediction, 2)

    return render_template("result.html", prediction_tip=prediction)

if __name__ == "__main__":
    app.run(debug=True)
