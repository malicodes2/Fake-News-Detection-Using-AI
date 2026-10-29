from datetime import datetime
from flask_login import UserMixin
from app import db, login_manager


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='user')  # 'user' or 'admin'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # relationships
    articles = db.relationship('NewsArticle', backref='author', lazy=True)
    datasets = db.relationship('Dataset', backref='uploader', lazy=True)

    def is_admin(self):
        return self.role == 'admin'

    def __repr__(self):
        return f'<User {self.username}>'


class NewsArticle(db.Model):
    __tablename__ = 'news_articles'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(300), nullable=True)
    content = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10))  # 'Real' or 'Fake'
    confidence = db.Column(db.Float)  # 0.0 to 100.0
    analyzed_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<NewsArticle {self.id} - {self.prediction}>'


class Dataset(db.Model):
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    row_count = db.Column(db.Integer, default=0)
    filepath = db.Column(db.String(500))

    def __repr__(self):
        return f'<Dataset {self.filename}>'


class ModelMetrics(db.Model):
    __tablename__ = 'model_metrics'

    id = db.Column(db.Integer, primary_key=True)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=True)
    training_samples = db.Column(db.Integer, default=0)

    dataset = db.relationship('Dataset', backref='metrics')

    def __repr__(self):
        return f'<ModelMetrics accuracy={self.accuracy:.2f}>'
