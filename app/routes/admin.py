import os
from functools import wraps

from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_required, current_user
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename

from app import db
from app.models import User, NewsArticle, Dataset, ModelMetrics
from app.forms import AdminUserForm

admin_bp = Blueprint('admin', __name__)


def admin_required(f):
    """Decorator to restrict access to admin users only."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    allowed = current_app.config.get('ALLOWED_EXTENSIONS', {'csv'})
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


# ---- admin dashboard ----
@admin_bp.route('/dashboard')
@login_required
@admin_required
def dashboard():
    total_users = User.query.count()
    total_articles = NewsArticle.query.count()
    total_datasets = Dataset.query.count()

    # get latest metrics if available
    latest_metrics = ModelMetrics.query.order_by(ModelMetrics.trained_at.desc()).first()

    return render_template('admin/dashboard.html',
                           total_users=total_users,
                           total_articles=total_articles,
                           total_datasets=total_datasets,
                           latest_metrics=latest_metrics)


# ---- user management ----
@admin_bp.route('/users')
@login_required
@admin_required
def users():
    all_users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users.html', users=all_users)


@admin_bp.route('/users/add', methods=['GET', 'POST'])
@login_required
@admin_required
def add_user():
    form = AdminUserForm()
    if form.validate_on_submit():
        # check duplicates
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists.', 'danger')
            return render_template('admin/user_form.html', form=form, action='Add')

        if User.query.filter_by(email=form.email.data.lower()).first():
            flash('Email already registered.', 'danger')
            return render_template('admin/user_form.html', form=form, action='Add')

        if not form.password.data:
            flash('Password is required for new users.', 'danger')
            return render_template('admin/user_form.html', form=form, action='Add')

        user = User(
            username=form.username.data.strip(),
            email=form.email.data.strip().lower(),
            password_hash=generate_password_hash(form.password.data),
            role=form.role.data
        )
        db.session.add(user)
        db.session.commit()
        flash(f'User "{user.username}" created successfully.', 'success')
        return redirect(url_for('admin.users'))

    return render_template('admin/user_form.html', form=form, action='Add')


@admin_bp.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    form = AdminUserForm()

    if form.validate_on_submit():
        # check if username taken by someone else
        dup = User.query.filter(User.username == form.username.data, User.id != user.id).first()
        if dup:
            flash('Username already taken.', 'danger')
            return render_template('admin/user_form.html', form=form, action='Edit', user=user)

        dup_email = User.query.filter(User.email == form.email.data.lower(), User.id != user.id).first()
        if dup_email:
            flash('Email already in use.', 'danger')
            return render_template('admin/user_form.html', form=form, action='Edit', user=user)

        user.username = form.username.data.strip()
        user.email = form.email.data.strip().lower()
        user.role = form.role.data

        # only update password if provided
        if form.password.data:
            user.password_hash = generate_password_hash(form.password.data)

        db.session.commit()
        flash(f'User "{user.username}" updated.', 'success')
        return redirect(url_for('admin.users'))

    # pre-fill form
    form.username.data = user.username
    form.email.data = user.email
    form.role.data = user.role
    return render_template('admin/user_form.html', form=form, action='Edit', user=user)


@admin_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)

    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin.users'))

    # delete user's articles too
    NewsArticle.query.filter_by(user_id=user.id).delete()
    db.session.delete(user)
    db.session.commit()
    flash(f'User "{user.username}" has been deleted.', 'success')
    return redirect(url_for('admin.users'))


# ---- dataset management ----
@admin_bp.route('/datasets', methods=['GET', 'POST'])
@login_required
@admin_required
def datasets():
    if request.method == 'POST':
        if 'dataset_file' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(url_for('admin.datasets'))

        file = request.files['dataset_file']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(url_for('admin.datasets'))

        if not allowed_file(file.filename):
            flash('Only CSV files are allowed.', 'danger')
            return redirect(url_for('admin.datasets'))

        filename = secure_filename(file.filename)
        save_path = os.path.join(current_app.config['DATASET_FOLDER'], filename)
        file.save(save_path)

        # count rows (use line count as fallback for messy CSVs)
        try:
            with open(save_path, 'r', encoding='utf-8', errors='ignore') as f:
                row_count = sum(1 for _ in f) - 1  # subtract header
                row_count = max(row_count, 0)
        except Exception:
            row_count = 0

        dataset = Dataset(
            filename=filename,
            uploaded_by=current_user.id,
            row_count=row_count,
            filepath=save_path
        )
        db.session.add(dataset)
        db.session.commit()
        flash(f'Dataset "{filename}" uploaded successfully ({row_count} rows).', 'success')
        return redirect(url_for('admin.datasets'))

    all_datasets = Dataset.query.order_by(Dataset.uploaded_at.desc()).all()
    return render_template('admin/datasets.html', datasets=all_datasets)


@admin_bp.route('/datasets/<int:dataset_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)

    # remove file from disk
    if dataset.filepath and os.path.exists(dataset.filepath):
        os.remove(dataset.filepath)

    # remove any associated metrics
    ModelMetrics.query.filter_by(dataset_id=dataset.id).delete()

    db.session.delete(dataset)
    db.session.commit()
    flash(f'Dataset "{dataset.filename}" deleted.', 'success')
    return redirect(url_for('admin.datasets'))


# ---- model retraining ----
@admin_bp.route('/retrain', methods=['POST'])
@login_required
@admin_required
def retrain():
    # get the latest dataset
    latest_dataset = Dataset.query.order_by(Dataset.uploaded_at.desc()).first()
    if not latest_dataset:
        flash('No dataset available. Please upload a dataset first.', 'danger')
        return redirect(url_for('admin.datasets'))

    try:
        from app.ml.model import train_model
        metrics = train_model(latest_dataset.filepath)

        # save metrics to database
        model_metrics = ModelMetrics(
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            dataset_id=latest_dataset.id,
            training_samples=metrics.get('training_samples', 0)
        )
        db.session.add(model_metrics)
        db.session.commit()

        flash(f'Model retrained successfully! Accuracy: {metrics["accuracy"]:.2%}', 'success')
    except Exception as e:
        flash(f'Error during retraining: {str(e)}', 'danger')

    return redirect(url_for('admin.metrics'))


# ---- performance metrics ----
@admin_bp.route('/metrics')
@login_required
@admin_required
def metrics():
    all_metrics = ModelMetrics.query.order_by(ModelMetrics.trained_at.desc()).all()
    latest = all_metrics[0] if all_metrics else None
    return render_template('admin/metrics.html', metrics_list=all_metrics, latest=latest)


# ---- usage reports ----
@admin_bp.route('/reports')
@login_required
@admin_required
def reports():
    total_users = User.query.filter_by(role='user').count()
    total_admins = User.query.filter_by(role='admin').count()
    total_articles = NewsArticle.query.count()
    total_real = NewsArticle.query.filter_by(prediction='Real').count()
    total_fake = NewsArticle.query.filter_by(prediction='Fake').count()
    total_datasets = Dataset.query.count()

    # per-user stats
    from sqlalchemy import func
    user_stats = db.session.query(
        User.username,
        func.count(NewsArticle.id).label('article_count')
    ).outerjoin(NewsArticle).group_by(User.id).all()

    return render_template('admin/reports.html',
                           total_users=total_users,
                           total_admins=total_admins,
                           total_articles=total_articles,
                           total_real=total_real,
                           total_fake=total_fake,
                           total_datasets=total_datasets,
                           user_stats=user_stats)

@admin_bp.route('/reports/export')
@login_required
@admin_required
def export_reports():
    import csv
    from io import StringIO
    from flask import Response

    total_users = User.query.filter_by(role='user').count()
    total_admins = User.query.filter_by(role='admin').count()
    total_articles = NewsArticle.query.count()
    total_real = NewsArticle.query.filter_by(prediction='Real').count()
    total_fake = NewsArticle.query.filter_by(prediction='Fake').count()
    total_datasets = Dataset.query.count()

    total_accuracy = "N/A"
    latest_metrics = ModelMetrics.query.order_by(ModelMetrics.trained_at.desc()).first()
    if latest_metrics:
        total_accuracy = f"{latest_metrics.accuracy * 100:.2f}%"

    si = StringIO()
    cw = csv.writer(si)
    
    # Write Header
    cw.writerow(['Metric', 'Value'])
    cw.writerow(['Total Registered Users', total_users])
    cw.writerow(['Total Administrators', total_admins])
    cw.writerow(['Total News Articles Analyzed', total_articles])
    cw.writerow(['Real News Detected', total_real])
    cw.writerow(['Fake News Detected', total_fake])
    cw.writerow(['Training Datasets Uploaded', total_datasets])
    cw.writerow(['Latest Model Accuracy', total_accuracy])
    cw.writerow([])
    
    # Write Per-User Activity
    cw.writerow(['User Activity Breakdown'])
    cw.writerow(['Username', 'Articles Analyzed'])
    
    from sqlalchemy import func
    user_stats = db.session.query(
        User.username,
        func.count(NewsArticle.id).label('article_count')
    ).outerjoin(NewsArticle).group_by(User.id).all()
    
    for stat in user_stats:
        cw.writerow([stat.username, stat.article_count])
        
    output = si.getvalue()
    
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=System_Usage_Report.csv"}
    )

