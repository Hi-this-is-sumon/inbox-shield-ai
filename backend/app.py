from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pickle
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

# Initialize App
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model/spam_model.pkl")
vectorizer_path = os.path.join(base_dir, "model/vectorizer.pkl")
whitelist_path = os.path.join(base_dir, "data/whitelist.csv")             # Personal Whitelist
trusted_domains_path = os.path.join(base_dir, "data/trusted_domains.csv") # Global Trusted Domains
landing_page_path = os.path.join(base_dir, "templates", "index.html")
static_dir = os.path.join(base_dir, "static")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Globals
model = None
vectorizer = None
personal_whitelist = set()
global_trusted_domains = set()
ps = PorterStemmer()
STOPWORDS = set(ENGLISH_STOP_WORDS)
CLEAN_REGEX = re.compile(r'[^a-z0-9\s$]')

def load_resources():
    global model, vectorizer, personal_whitelist, global_trusted_domains

    # 1. Load ML Model
    try:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, "rb") as model_file, open(vectorizer_path, "rb") as vectorizer_file:
                model = pickle.load(model_file)
                vectorizer = pickle.load(vectorizer_file)
        else:
            print("WARNING: Model/vectorizer not found.")
    except Exception as exc:
        print(f"Error loading model: {exc}")

    # 2. Load Personal Whitelist (Emails & Domains you explicitly trust)
    try:
        if os.path.exists(whitelist_path):
            df_white = pd.read_csv(whitelist_path)
            if 'email' in df_white.columns:
                personal_whitelist.update(df_white['email'].dropna().astype(str).str.lower().values)
            if 'domain' in df_white.columns:
                # Add whitelisted domains to the global trusted list
                global_trusted_domains.update(df_white['domain'].dropna().astype(str).str.lower().values)
    except Exception as exc:
        print(f"Error loading whitelist.csv: {exc}")

    # 3. Load Global Trusted Domains (Banks, Edu, Enterprise)
    try:
        if os.path.exists(trusted_domains_path):
            df_trust = pd.read_csv(trusted_domains_path, header=None)
            global_trusted_domains.update(df_trust[0].dropna().astype(str).str.lower().values)
            print(f"Loaded {len(global_trusted_domains)} trusted domains.")
    except Exception as exc:
        print(f"Error loading trusted_domains.csv: {exc}")

load_resources()

class EmailRequest(BaseModel):
    sender: str
    subject: str
    body: str

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text[:3000].lower()
    text = CLEAN_REGEX.sub(' ', text)
    text = [ps.stem(word) for word in text.split() if word not in STOPWORDS]
    return ' '.join(text)

def check_whitelist(sender):
    """Layer 1: Check against explicitly whitelisted personal emails."""
    if not sender: return False
    return sender.lower() in personal_whitelist

def is_trusted_domain(sender):
    """Layer 2: Check against massive CSV of known safe domains securely."""
    if not sender or '@' not in sender: return False
    domain = sender.split('@')[-1].lower()
    
    for trusted in global_trusted_domains:
        if domain == trusted or domain.endswith('.' + trusted):
            return True
    return False

def has_spam_indicators(subject, body):
    """Layer 2.5: Expanded list of modern scam triggers."""
    text = f"{subject} {body}".lower()
    
    spam_keywords = [
        # Financial / Crypto Scams
        'bitcoin', 'cryptocurrency', 'crypto wallet', 'guaranteed returns', 'investment scheme',
        # Urgency / Phishing
        'urgent act', 'limited time', 'act now', 'expire soon', 'verify your account immediately',
        'suspended account', 'confirm identity', 'unauthorized login', 'account restricted',
        'update kyc', 'kyc pending', 'bank account blocked',
        # Prizes / Greed
        'winner', 'lottery', 'prize', 'claim now', 'congratulations!!!', 'free money', 'wire transfer'
    ]
    
    spam_count = sum(1 for keyword in spam_keywords if keyword in text)
    return spam_count >= 2

@app.get("/")
def home():
    if os.path.exists(landing_page_path): return FileResponse(landing_page_path)
    return {"message": "API is running. Landing page missing."}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "trusted_domains_count": len(global_trusted_domains)}

@app.post("/predict")
def predict_spam(email: EmailRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Layer 1: Personal Whitelist
    if check_whitelist(email.sender):
        return {"label": "Not Spam", "confidence": 1.0, "reason": "Sender is in your personal whitelist."}
    
    # Layer 2: Global Trusted Domains
    if is_trusted_domain(email.sender):
        return {"label": "Not Spam", "confidence": 0.95, "reason": "Recognized as a globally trusted domain."}
    
    # Layer 2.5: Obvious Spam Indicators
    if has_spam_indicators(email.subject, email.body):
        return {"label": "Spam", "confidence": 0.95, "reason": "Contains multiple severe phishing/spam indicators."}

    # Layer 3: ML Model Prediction
    full_text = f"{email.subject} {email.body}"
    processed_text = preprocess_text(full_text)
    vectorized_text = vectorizer.transform([processed_text])
    
    proba = model.predict_proba(vectorized_text)[0]
    spam_probability = float(proba[1])
    
    if spam_probability >= 0.5:
        return {
            "label": "Spam",
            "confidence": round(spam_probability, 2),
            "reason": "AI detected suspicious language patterns.",
            "analysis": f"High probability of spam ({spam_probability:.1%})."
        }
    else:
        return {
            "label": "Not Spam",
            "confidence": round(float(proba[0]), 2),
            "reason": "AI analysis looks normal.",
            "analysis": f"Appears legitimate ({float(proba[0]):.1%})."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)