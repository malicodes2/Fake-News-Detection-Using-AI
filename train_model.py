"""
Download the Kaggle Fake and Real News Dataset and train the model.
Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
"""
import os
import pandas as pd
import kagglehub

# download dataset
print("Downloading Fake and Real News Dataset from Kaggle...")
print("(This may ask for Kaggle credentials on first run)\n")

path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
print(f"Dataset downloaded to: {path}")

# list files in the download folder
files = os.listdir(path)
print(f"Files found: {files}")

# load and combine the two CSV files (True.csv and Fake.csv)
true_df = pd.read_csv(os.path.join(path, 'True.csv'))
fake_df = pd.read_csv(os.path.join(path, 'Fake.csv'))

print(f"\nTrue news articles: {len(true_df)}")
print(f"Fake news articles: {len(fake_df)}")

# add labels
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# combine into single dataset
combined = pd.concat([true_df, fake_df], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# keep relevant columns
combined = combined[['title', 'text', 'label']]

# save combined CSV to uploads/datasets
output_dir = os.path.join(os.path.dirname(__file__), 'uploads', 'datasets')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'fake_news_dataset.csv')
combined.to_csv(output_path, index=False)

print(f"\nCombined dataset saved to: {output_path}")
print(f"Total rows: {len(combined)}")
print(f"Columns: {list(combined.columns)}")
print(f"\nLabel distribution:")
print(combined['label'].value_counts())

# now train the model
print("\n" + "=" * 50)
print("TRAINING THE MODEL...")
print("=" * 50)

from app.ml.model import train_model

metrics = train_model(output_path)

print(f"\nTraining complete!")
print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.1f}%)")
print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.1f}%)")
print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.1f}%)")
print(f"  Training samples: {metrics['training_samples']}")

# also save metrics to database
print("\nSaving metrics to database...")
from app import create_app, db
from app.models import Dataset, ModelMetrics

app = create_app()
with app.app_context():
    # save dataset record
    ds = Dataset(
        filename='fake_news_dataset.csv',
        uploaded_by=1,  # admin user
        row_count=len(combined),
        filepath=output_path
    )
    db.session.add(ds)
    db.session.commit()

    # save metrics
    mm = ModelMetrics(
        accuracy=metrics['accuracy'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        f1_score=metrics['f1_score'],
        dataset_id=ds.id,
        training_samples=metrics['training_samples']
    )
    db.session.add(mm)
    db.session.commit()

    print("Metrics saved to database!")

print("\n" + "=" * 50)
print("ALL DONE! The model is ready to use.")
print("Run 'python run.py' and try analyzing some news articles.")
print("=" * 50)
