from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
import joblib
import pandas as pd
import json
from database import db, User, Prediction

app = Flask(__name__)
app.config['SECRET_KEY'] = 'healthlyzer-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///healthlyzer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)

import json as _json
@app.template_filter('from_json')
def from_json_filter(s):
    try:
        return _json.loads(s)
    except Exception:
        return []

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

model = joblib.load("model.pkl")

SYMPTOM_COLS = [
    'fever', 'cough', 'headache', 'vomiting', 'fatigue',
    'body_ache', 'sore_throat', 'runny_nose', 'chest_pain',
    'shortness_of_breath', 'skin_rash', 'loss_of_taste',
    'diarrhea', 'chills', 'muscle_weakness'
]

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─────────────────────────────────────────────
#  HOME / ANALYZER
# ─────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    active_symptoms = []

    if request.method == "POST":
        if "agree" not in request.form:
            flash("You must agree to the Terms & Conditions first.", "error")
            return render_template("index.html", prediction=None, symptom_cols=SYMPTOM_COLS)

        values = {col: (1 if request.form.get(col) == "1" else 0) for col in SYMPTOM_COLS}
        active_symptoms = [col for col, v in values.items() if v == 1]

        input_data = pd.DataFrame([list(values.values())], columns=SYMPTOM_COLS)
        prediction = model.predict(input_data)[0]

        # Save to history if logged in
        if current_user.is_authenticated:
            entry = Prediction(
                user_id=current_user.id,
                symptoms=json.dumps(active_symptoms),
                result=prediction
            )
            db.session.add(entry)
            db.session.commit()

    return render_template("index.html", prediction=prediction,
                           symptom_cols=SYMPTOM_COLS, active_symptoms=active_symptoms)

# ─────────────────────────────────────────────
#  AUTH
# ─────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")

        if not name or not email or not password:
            flash("All fields are required.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
        elif User.query.filter_by(email=email).first():
            flash("An account with this email already exists.", "error")
        else:
            hashed = bcrypt.generate_password_hash(password).decode('utf-8')
            user = User(name=name, email=email, password=hashed)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            flash(f"Welcome to Healthlyzer, {name}! 🎉", "success")
            return redirect(url_for('home'))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=request.form.get("remember") == "on")
            flash(f"Welcome back, {user.name}!", "success")
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash("Invalid email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for('home'))

# ─────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                  .order_by(Prediction.timestamp.desc()).all()
    # Chart data
    disease_counts = {}
    for p in predictions:
        disease_counts[p.result] = disease_counts.get(p.result, 0) + 1

    chart_labels = list(disease_counts.keys())
    chart_data   = list(disease_counts.values())

    return render_template("dashboard.html",
                           predictions=predictions,
                           disease_counts=disease_counts,
                           chart_labels=json.dumps(chart_labels),
                           chart_data=json.dumps(chart_data))


@app.route("/dashboard/clear", methods=["POST"])
@login_required
def clear_history():
    Prediction.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash("Prediction history cleared.", "info")
    return redirect(url_for('dashboard'))

# ─────────────────────────────────────────────
#  PROFILE
# ─────────────────────────────────────────────
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        action = request.form.get("action")

        if action == "update_name":
            name = request.form.get("name", "").strip()
            if not name:
                flash("Name cannot be empty.", "error")
            else:
                current_user.name = name
                db.session.commit()
                flash("Name updated successfully.", "success")

        elif action == "change_password":
            current_pw  = request.form.get("current_password", "")
            new_pw      = request.form.get("new_password", "")
            confirm_pw  = request.form.get("confirm_password", "")
            if not bcrypt.check_password_hash(current_user.password, current_pw):
                flash("Current password is incorrect.", "error")
            elif new_pw != confirm_pw:
                flash("New passwords do not match.", "error")
            elif len(new_pw) < 6:
                flash("Password must be at least 6 characters.", "error")
            else:
                current_user.password = bcrypt.generate_password_hash(new_pw).decode('utf-8')
                db.session.commit()
                flash("Password changed successfully.", "success")

    return render_template("profile.html")

# ─────────────────────────────────────────────
#  ABOUT
# ─────────────────────────────────────────────
@app.route("/about")
def about():
    return render_template("about.html")

# ─────────────────────────────────────────────
#  LEGAL
# ─────────────────────────────────────────────
@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
