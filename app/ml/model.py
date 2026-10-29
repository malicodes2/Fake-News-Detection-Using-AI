import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.ml.preprocessing import clean_text, extract_features

# paths for saving model artifacts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'app', 'ml', 'saved_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'classifier.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')


def train_model(dataset_path):
    """
    Train the fake news classifier on a CSV dataset.
    Expects columns: 'text' (or 'title'), and 'label' (REAL/FAKE or 1/0).
    Returns dict with accuracy, precision, recall, f1_score.
    """
    # try utf-8 first, fall back to latin-1 for datasets with special characters
    try:
        df = pd.read_csv(dataset_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding='latin-1')

    # try to figure out the right column names
    # common datasets have different column naming
    text_col = None
    label_col = None

    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ('text', 'article', 'content', 'news'):
            text_col = col
        elif col_lower in ('title', 'headline') and text_col is None:
            text_col = col
        elif col_lower in ('label', 'class', 'target', 'category'):
            label_col = col

    if text_col is None or label_col is None:
        raise ValueError(
            f'Could not find text/label columns. Found: {list(df.columns)}. '
            f'Expected columns like "text" and "label".'
        )

    # drop rows with missing values
    df = df[[text_col, label_col]].dropna()

    # normalize labels to 'Real' and 'Fake'
    label_mapping = {}
    unique_labels = df[label_col].unique()
    for lbl in unique_labels:
        lbl_str = str(lbl).strip().upper()
        if lbl_str in ('REAL', 'TRUE', '1', 'RELIABLE'):
            label_mapping[lbl] = 'Real'
        elif lbl_str in ('FAKE', 'FALSE', '0', 'UNRELIABLE'):
            label_mapping[lbl] = 'Fake'

    if len(label_mapping) < 2:
        # fallback: treat first unique as Real, second as Fake
        labels_list = list(unique_labels)
        if len(labels_list) >= 2:
            label_mapping[labels_list[0]] = 'Real'
            label_mapping[labels_list[1]] = 'Fake'
        else:
            raise ValueError('Dataset must have at least 2 distinct labels.')

    df['clean_label'] = df[label_col].map(label_mapping)
    df = df.dropna(subset=['clean_label'])

    # DEBUG: show label distribution
    print(f"[TRAIN DEBUG] Text column: '{text_col}', Label column: '{label_col}'")
    print(f"[TRAIN DEBUG] Label mapping: {label_mapping}")
    print(f"[TRAIN DEBUG] Label distribution:\n{df['clean_label'].value_counts().to_string()}")

    # preprocess text and calculate linguistic features
    df['processed'] = df[text_col].apply(clean_text)
    df['features'] = df[text_col].apply(extract_features)

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df[['processed', 'features']], df['clean_label'],
        test_size=0.2, random_state=42
    )

    # vectorize the text data using TF-IDF
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words='english',
        sublinear_tf=True,
        min_df=3,
        max_df=0.95
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train['processed'])
    X_test_tfidf = tfidf.transform(X_test['processed'])
    
    # extract the numeric features into a usable matrix
    numeric_keys = ['lexical_diversity', 'avg_word_length', 'punctuation_ratio']
    
    def dicts_to_matrix(feature_series):
        matrix = []
        for d in feature_series:
            row = [d.get(k, 0.0) for k in numeric_keys]
            matrix.append(row)
        return np.array(matrix)
        
    X_train_num = dicts_to_matrix(X_train['features'])
    X_test_num = dicts_to_matrix(X_test['features'])
    
    # combine the TF-IDF matrix with the numeric features matrix
    X_train_combined = hstack([X_train_tfidf, X_train_num])
    X_test_combined = hstack([X_test_tfidf, X_test_num])

    # train the Naive Bayes model
    base_clf = MultinomialNB(alpha=1.0)
    calibrated_clf = CalibratedClassifierCV(base_clf, cv=5)
    calibrated_clf.fit(X_train_combined, y_train)

    # DEBUG: show model info
    # Note: calibrated_clf holds the classes
    print(f"[TRAIN DEBUG] Model classes: {calibrated_clf.classes_}")
    print(f"[TRAIN DEBUG] Train distribution:\n{y_train.value_counts().to_string()}")

    # evaluate
    y_pred = calibrated_clf.predict(X_test_combined)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='Fake', average='binary')
    rec = recall_score(y_test, y_pred, pos_label='Fake', average='binary')
    f1 = f1_score(y_test, y_pred, pos_label='Fake', average='binary')

    # save the trained model components
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(calibrated_clf, MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'training_samples': len(X_train)
    }


def predict_news(text):
    """
    Predict whether a news article is Real or Fake.
    Returns dict with 'label' and 'confidence' (percentage).
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('Model not trained yet. Please ask admin to train the model.')

    # load the saved model and vectorizer
    classifier = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    
    # clean the input text and extract features
    cleaned = clean_text(text)
    features = extract_features(text)
    
    vectorized_text = tfidf.transform([cleaned])
    
    numeric_keys = ['lexical_diversity', 'avg_word_length', 'punctuation_ratio']
    numeric_arr = np.array([[features.get(k, 0.0) for k in numeric_keys]])
    
    # combine features for prediction
    combined_input = hstack([vectorized_text, numeric_arr])

    # get prediction and calculate confidence percentage
    prediction = classifier.predict(combined_input)[0]
    proba = classifier.predict_proba(combined_input)[0]

    confidence = max(proba) * 100

    print(f"[PREDICT DEBUG] Classes: {classifier.classes_}, Probabilities: {proba}")
    print(f"[PREDICT DEBUG] Prediction: {prediction}, Confidence: {confidence:.1f}%")

    return {
        'label': prediction,
        'confidence': round(confidence, 1)
    }
