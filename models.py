from flask_sqlalchemy import SQLAlchemy

# Database instance will be initialized by web_server.py
db = SQLAlchemy()


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    company = db.Column(db.String(255))
    role = db.Column(db.String(20), nullable=False, default='viewer') # 'operator' or 'viewer'
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    
    # Virtual property for full name
    @property
    def name(self):
        return f"{self.first_name} {self.last_name}"


class RefreshToken(db.Model):
    __tablename__ = 'refresh_tokens'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    token_hash = db.Column(db.String(255), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    user = db.relationship('User', backref=db.backref('refresh_tokens', lazy=True))


class KnownFace(db.Model):
    __tablename__ = 'known_faces'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


class FaceProfile(db.Model):
    __tablename__ = 'face_profiles'

    id = db.Column(db.Integer, primary_key=True)
    known_face_id = db.Column(db.Integer, db.ForeignKey('known_faces.id'), nullable=False)
    label = db.Column(db.String(120))
    image_path = db.Column(db.String(255))
    face_encoding = db.Column(db.LargeBinary)
    known_face = db.relationship('KnownFace', backref=db.backref('face_profiles', lazy=True))


class Session(db.Model):
    __tablename__ = 'sessions'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    started_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
    ended_at = db.Column(db.DateTime)


class DetectionEvent(db.Model):
    __tablename__ = 'detection_events'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    known_face_id = db.Column(db.Integer, db.ForeignKey('known_faces.id'))
    timestamp = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
    result = db.Column(db.String(20), nullable=False)  # e.g., "recognized" or "unknown"
    confidence = db.Column(db.Float)
    session = db.relationship('Session', backref=db.backref('detection_events', lazy=True))
    known_face = db.relationship('KnownFace', backref=db.backref('detection_events', lazy=True))


class Command(db.Model):
    __tablename__ = 'commands'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.id'), nullable=False)
    issued_at = db.Column(db.DateTime, server_default=db.func.now(), nullable=False)
    command = db.Column(db.String(255), nullable=False)
    session = db.relationship('Session', backref=db.backref('commands', lazy=True))