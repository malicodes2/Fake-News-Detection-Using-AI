import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'fn-detect-dev-key-2025'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'instance', 'fakenews.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # file upload settings
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
    DATASET_FOLDER = os.path.join(basedir, 'uploads', 'datasets')
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200MB max upload

    # ML model paths
    MODEL_FOLDER = os.path.join(basedir, 'app', 'ml', 'saved_model')
    ALLOWED_EXTENSIONS = {'csv'}
