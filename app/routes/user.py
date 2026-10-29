from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user

from app import db
from app.models import User, NewsArticle
from app.forms import ProfileForm, NewsSubmitForm

user_bp = Blueprint('user', __name__)


@user_bp.route('/dashboard')
@login_required
def dashboard():
    # grab recent articles for this user
    recent_articles = NewsArticle.query.filter_by(user_id=current_user.id)\
        .order_by(NewsArticle.analyzed_at.desc()).limit(5).all()
    total_articles = NewsArticle.query.filter_by(user_id=current_user.id).count()
    real_count = NewsArticle.query.filter_by(user_id=current_user.id, prediction='Real').count()
    fake_count = NewsArticle.query.filter_by(user_id=current_user.id, prediction='Fake').count()

    return render_template('user/dashboard.html',
                           recent_articles=recent_articles,
                           total_articles=total_articles,
                           real_count=real_count,
                           fake_count=fake_count)


@user_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = ProfileForm()

    if form.validate_on_submit():
        # check if username/email already taken by someone else
        existing_user = User.query.filter(
            User.username == form.username.data,
            User.id != current_user.id
        ).first()
        if existing_user:
            flash('Username already taken.', 'danger')
            return render_template('user/profile.html', form=form)

        existing_email = User.query.filter(
            User.email == form.email.data.lower(),
            User.id != current_user.id
        ).first()
        if existing_email:
            flash('Email already in use.', 'danger')
            return render_template('user/profile.html', form=form)

        current_user.username = form.username.data.strip()
        current_user.email = form.email.data.strip().lower()
        db.session.commit()
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('user.profile'))

    # pre-fill form with current data
    form.username.data = current_user.username
    form.email.data = current_user.email
    return render_template('user/profile.html', form=form)


@user_bp.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    form = NewsSubmitForm()

    if form.validate_on_submit():
        text = form.content.data.strip()
        title = form.title.data.strip() if form.title.data else None

        # auto-generate title from content if not provided
        if not title:
            words = text.split()[:8]
            title = ' '.join(words)
            if len(text.split()) > 8:
                title += '…'

        # run the ML prediction
        try:
            from app.ml.model import predict_news
            result = predict_news(text)
            prediction = result['label']
            confidence = result['confidence']
        except Exception as e:
            flash(f'Model not available. Please ask admin to train the model first. ({str(e)})', 'warning')
            return render_template('user/analyze.html', form=form)

        # save to database
        article = NewsArticle(
            user_id=current_user.id,
            title=title,
            content=text,
            prediction=prediction,
            confidence=round(confidence, 2)
        )
        db.session.add(article)
        db.session.commit()

        return render_template('user/result.html', article=article, result=result)

    return render_template('user/analyze.html', form=form)


@user_bp.route('/history')
@login_required
def history():
    articles = NewsArticle.query.filter_by(user_id=current_user.id)\
        .order_by(NewsArticle.analyzed_at.desc()).all()
    return render_template('user/history.html', articles=articles)


@user_bp.route('/article/<int:article_id>')
@login_required
def article_detail(article_id):
    article = NewsArticle.query.get_or_404(article_id)
    # only allow owner to view their own article
    if article.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('user.history'))
    result = {
        'label': article.prediction,
        'confidence': article.confidence
    }
    return render_template('user/article_detail.html', article=article, result=result)
