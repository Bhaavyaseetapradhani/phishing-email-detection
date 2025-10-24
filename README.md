# AI-Powered Phishing Email Detection

##  Project Overview
Machine learning classifier that detects phishing emails using NLP feature extraction and Random Forest algorithm. Achieves 92% precision and 90% recall on validation data.

**Status**:  Production Ready

##  Key Metrics
| Metric | Value |
|--------|-------|
| **ROC-AUC Score** | 0.92+ |
| **F1-Score** | 0.91 |
| **Precision** | 92% |
| **Recall** | 90% |

##  Files
- `phishing_detector.py` - Main ML classifier
- `requirements.txt` - Python dependencies
- `sample_data.csv` - Training dataset
- `README.md` - This file

##  Quick Start

### 1. Install Python Libraries
```bash
pip install -r requirements.txt
```

### 2. Run the Detector
```bash
python phishing_detector.py
```

##  How It Works
- Uses Random Forest classifier (100 trees)
- Extracts text statistics (character count, uppercase ratio)
- Applies TF-IDF vectorization (1000 features)
- 5-fold cross-validation for validation

##  Test with Your Own Email
Edit `phishing_detector.py` and change:
```python
test_email = "Your email text here"
```

---
**Author**: Bhaavya Seeta Pradhani
**Status**:  Production Ready
