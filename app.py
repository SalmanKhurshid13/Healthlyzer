from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        if "agree" not in request.form:
            return render_template("index.html", prediction="You must agree to the Terms & Conditions first.")
        fever    = 1 if request.form.get("fever")    == "1" else 0
        cough    = 1 if request.form.get("cough")    == "1" else 0
        headache = 1 if request.form.get("headache") == "1" else 0
        vomiting = 1 if request.form.get("vomiting") == "1" else 0
        fatigue  = 1 if request.form.get("fatigue")  == "1" else 0
        input_data = pd.DataFrame(
            [[fever, cough, headache, vomiting, fatigue]],
            columns=["fever", "cough", "headache", "vomiting", "fatigue"]
        )
        prediction = model.predict(input_data)[0]
    return render_template("index.html", prediction=prediction)

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

if __name__ == "__main__":
    app.run(debug=True)