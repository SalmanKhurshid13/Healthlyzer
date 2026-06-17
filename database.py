from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(100), nullable=False)
    email        = db.Column(db.String(150), unique=True, nullable=False)
    password     = db.Column(db.String(200), nullable=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    predictions  = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')

    def prediction_count(self):
        return len(self.predictions)

    def most_common_disease(self):
        if not self.predictions:
            return None
        counts = {}
        for p in self.predictions:
            counts[p.result] = counts.get(p.result, 0) + 1
        return max(counts, key=counts.get)


class Prediction(db.Model):
    __tablename__ = 'predictions'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symptoms   = db.Column(db.Text, nullable=False)   # JSON list of active symptom names
    result     = db.Column(db.String(100), nullable=False)
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)

    def symptoms_list(self):
        try:
            return json.loads(self.symptoms)
        except Exception:
            return []

    def symptoms_display(self):
        return ', '.join(s.replace('_', ' ').title() for s in self.symptoms_list()) or 'None'
