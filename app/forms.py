from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, TextAreaField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError

from app.models import User


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(), Length(min=3, max=80)
    ])
    email = StringField('Email', validators=[
        DataRequired(), Email()
    ])
    password = PasswordField('Password', validators=[
        DataRequired(), Length(min=6, message='Password must be at least 6 characters')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(), EqualTo('password', message='Passwords do not match')
    ])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username already taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email already registered.')


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


class ProfileForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(), Length(min=3, max=80)
    ])
    email = StringField('Email', validators=[
        DataRequired(), Email()
    ])
    submit = SubmitField('Update Profile')


class NewsSubmitForm(FlaskForm):
    title = StringField('Headline / Title', validators=[Length(max=300)])
    content = TextAreaField('News Article Text', validators=[
        DataRequired(), Length(min=10, message='Please enter at least 10 characters')
    ])
    submit = SubmitField('Analyze')


class AdminUserForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(), Length(min=3, max=80)
    ])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password')  # optional for updates
    role = SelectField('Role', choices=[('user', 'User'), ('admin', 'Admin')])
    submit = SubmitField('Save')
