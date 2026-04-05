import pandas as pd
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Download NLTK data quietly
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# Instantiate stemmer once globally to save processing time
ps = PorterStemmer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Truncate, lowercase, and keep only letters/numbers/spaces/$
    text = text[:3000].lower()
    text = re.sub(r'[^a-z0-9\s$]', ' ', text)
    
    # Stem and remove stopwords
    text = [ps.stem(word) for word in text.split() if word not in STOPWORDS]
    
    return ' '.join(text)

def train():
    print("DEBUG: Running optimized train_model.py")
    
    # Get script directory (unchanged to preserve your paths)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'spam.csv')
    model_path = os.path.join(base_dir, 'spam_model.pkl')
    vectorizer_path = os.path.join(base_dir, 'vectorizer.pkl')

    print(f"Loading dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path, encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: spam.csv not found at {data_path}")
        return

    # Standardize columns
    if 'Body' in df.columns and 'Label' in df.columns:
        df = df[['Body', 'Label']].rename(columns={'Body': 'message', 'Label': 'label'})
    elif 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v2', 'v1']].rename(columns={'v2': 'message', 'v1': 'label'})
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    elif 'msg' in df.columns and 'label' in df.columns:
        df = df[['msg', 'label']].rename(columns={'msg': 'message'})
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"Dataset shape: {df.shape}")
    
    print("Preprocessing data...")
    df.dropna(subset=['message', 'label'], inplace=True)
    df['message'] = df['message'].apply(preprocess_text)
    df['label'] = df['label'].astype(int)

    # 1. SPLIT FIRST to prevent data leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.20, random_state=42
    )

    # 2. OVERSAMPLE ONLY THE TRAINING SET
    print("Balancing training data...")
    train_df = pd.DataFrame({'message': X_train_raw, 'label': y_train})
    spam_train = train_df[train_df['label'] == 1]
    ham_train = train_df[train_df['label'] == 0]
    
    spam_upsampled = spam_train.sample(n=len(ham_train), replace=True, random_state=42)
    train_balanced = pd.concat([ham_train, spam_upsampled])
    
    X_train_bal = train_balanced['message']
    y_train_bal = train_balanced['label']

    print(f"Class distribution after training set balancing:\n{y_train_bal.value_counts()}")

    # 3. VECTORIZE (Fit on train, transform on test)
    print("Vectorizing text with N-grams...")
    cv = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Removed .toarray() to preserve sparse matrix and save RAM
    X_train_vec = cv.fit_transform(X_train_bal)
    X_test_vec = cv.transform(X_test_raw)

    # 4. TRAIN
    print("Training MultinomialNB model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train_bal)

    # 5. EVALUATE
    y_pred = model.predict(X_test_vec)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 6. SAVE safely
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as m_file, open(vectorizer_path, 'wb') as v_file:
        pickle.dump(model, m_file)
        pickle.dump(cv, v_file)
    print("Done!")

if __name__ == "__main__":
    train()