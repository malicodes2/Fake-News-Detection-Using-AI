from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import os

from config import Config

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # make sure required dirs exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.instance_path), exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    # register blueprints
    from app.routes.auth import auth_bp
    from app.routes.user import user_bp
    from app.routes.admin import admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(admin_bp, url_prefix='/admin')

    # create tables on first run
    with app.app_context():
        from app import models
        db.create_all()

        # create default admin if none exists
        _create_default_admin(app)

    return app


def _create_default_admin(app):
    """Seed a default admin account so the system is usable right away."""
    from app.models import User
    from werkzeug.security import generate_password_hash

    admin = User.query.filter_by(role='admin').first()
    if admin is None:
        admin = User(
            username='admin',
            email='admin@fakenews.com',
            password_hash=generate_password_hash('admin123'),
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()
