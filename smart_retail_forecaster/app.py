from flask import Flask, render_template, request
import pickle
import numpy as np
import datetime

app = Flask(__name__)

# Load trained model
with open("model\demand_model.pkl", "rb") as f:
    model = pickle.load(f)
    
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        date_str = request.form["date"]
        is_promo = int(request.form["is_promo"])

        # Convert date
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        day = date_obj.day
        month = date_obj.month
        year = date_obj.year

        # Prepare features
        features = np.array([[day, month, year, is_promo]])
        prediction = model.predict(features)[0]

        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)