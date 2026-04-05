# Inbox Shield AI

AI-powered spam detection system for Gmail with modern animated Chrome extension & Python ML backend.

**Live Site / API:** [https://inbox-shield-ai-lake.vercel.app/](https://inbox-shield-ai-lake.vercel.app/)

**Last Updated:** April 05, 2026

## Features
- Advanced animated SVG shield icon
- 4-Layer Detection: Whitelist + Rules + Keywords + ML
- Real-time Gmail auto-scan
- 50% threshold MultinomialNB
- Recognizes 30+ financial services

## Quick Start
1. Visit [site](https://inbox-shield-ai-lake.vercel.app/)
2. Download ZIP → Load unpacked in chrome://extensions/
3. Pin & use in Gmail

## Tech Stack
**Frontend**: Manifest V3, 50+ CSS animations
**Backend**: FastAPI, scikit-learn, NLTK, TF-IDF (5000 features)

## Developer Setup
```bash
git clone https://github.com/Hi-this-is-sumon/inbox-shield-ai.git
cd inbox-shield-ai/backend
python -m venv venv && venv\\Scripts\\activate
pip install -r requirements.txt
python model/train_model.py
python app.py
```

## Performance (Latest Training)
- **100% Test Accuracy** (20k samples, F1=1.00)
- **80% Verification** (8/10 edge cases: spam/ham/phishing)
- 50% production threshold

## Structure
```
Inbox-Shield-AI/
├── backend/ (FastAPI/ML)
├── extension/ (Chrome MV3)
├── api/ (Vercel)
└── ...
```

MIT License | Sumon Mandal

[![Deploy](https://vercel.com/button)](https://vercel.com/new/git/external?repository-url=https://github.com/Hi-this-is-sumon/inbox-shield-ai)
