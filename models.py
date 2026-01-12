from __init__ import db

class Person(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    faces = db.relationship('Face', backref='person', lazy=True, cascade="all, delete-orphan")

class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('person.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    face_encoding = db.Column(db.LargeBinary)
    confidence = db.Column(db.Float)
