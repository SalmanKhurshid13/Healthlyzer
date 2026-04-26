from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        fever = int(request.form["fever"])
        cough = int(request.form["cough"])
        headache = int(request.form["headache"])
        vomiting = int(request.form["vomiting"])
        fatigue = int(request.form["fatigue"])

        input_data = pd.DataFrame(
            [[fever, cough, headache, vomiting, fatigue]],
            columns=['fever', 'cough', 'headache', 'vomiting', 'fatigue']
        )

        prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)