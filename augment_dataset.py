import os
import pandas as pd
from datasets import load_dataset
import kagglehub

def add_ai_dataset_to_kaggle():
    print("Downloading Kaggle Fake and Real News Dataset (in case it is missing)...")
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    
    true_df = pd.read_csv(os.path.join(path, 'True.csv'))
    fake_df = pd.read_csv(os.path.join(path, 'Fake.csv'))
    true_df['label'] = 'REAL' # Human
    fake_df['label'] = 'FAKE' # Human Fake
    
    combined_original = pd.concat([true_df, fake_df], ignore_index=True)[['title', 'text', 'label']]
    print(f"Original Dataset Size: {len(combined_original)}")

    # define some templates for synthetic AI-generated fake news
    ai_templates = [
        "A controversial government pilot program deploying artificial intelligence systems to handle administrative services has reduced processing times by nearly 35 percent, according to an internal performance review released Tuesday.",
        "A new industry report claims that artificial intelligence productivity tools are increasing workplace efficiency by nearly 30 percent — but researchers warn the gains may come at a hidden cognitive cost.",
        "Move over designer dogs — a growing number of celebrities are trading leashes and litter boxes for charging docks. Artificial intelligence–powered companion pets are quickly becoming the latest luxury accessory among A-list stars.",
        "According to a comprehensive study by Stanford researchers, regular consumption of highly synthesized laboratory water improves memory retention by approximately forty-two percent in adults over fifty.",
        "Officials announced today that the highly anticipated infrastructure bill will allocate over ten billion dollars exclusively toward the construction of autonomous vehicle lanes on all major interstate highways, starting next spring.",
        "A breakthrough in quantum mechanics at the European Organization for Nuclear Research has reportedly enabled scientists to successfully transmit particles backward in time by fractions of a nanosecond, though the findings remain under peer review."
    ]
    
    # generate a bunch of fake news samples based on the templates
    ai_texts = []
    for i in range(1000):
        for template in ai_templates:
            varied = template.replace("Tuesday", "Wednesday" if i%2==0 else "Monday")
            varied = varied.replace("percent", "per cent" if i%3==0 else "percent")
            ai_texts.append({
                'title': 'AI Generated News',
                'text': varied + f" Furthermore, experts indicate that consequence {i} could follow.",
                'label': 'FAKE'
            })
            
    ai_subset = pd.DataFrame(ai_texts)
    print(f"Adding {len(ai_subset)} AI-generated Formal Fake News Samples...")
    
    # combine everything and shuffle
    final_combined = pd.concat([combined_original, ai_subset], ignore_index=True)
    final_combined = final_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'uploads', 'datasets')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fake_news_augmented_dataset.csv')
    
    # clean up missing data and save
    final_combined = final_combined.dropna(subset=['text', 'label'])
    
    final_combined.to_csv(output_path, index=False)
    print(f"Augmented dataset saved to: {output_path}")
    print(f"Total rows: {len(final_combined)}")
    print(final_combined['label'].value_counts())

if __name__ == "__main__":
    add_ai_dataset_to_kaggle()
