"""
AI-Powered Phishing Email Detection System
Uses machine learning to detect phishing emails
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class PhishingEmailDetector:
    """Machine learning classifier for phishing email detection"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, 
            ngram_range=(1,2), 
            min_df=2, 
            max_df=0.95, 
            stop_words='english'
        )
        self.model = None
        self.metrics = {}
        
    def extract_text_features(self, text):
        """Extract simple features from email text"""
        if not isinstance(text, str):
            text = str(text)
        
        features = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'digit_ratio': sum(1 for c in text if c.isdigit()) / max(len(text), 1),
        }
        return features
    
    def preprocess_text(self, text):
        """Clean text for TF-IDF"""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        import re
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text
    
    def prepare_data(self, df, text_column='email_body', label_column='is_phishing'):
        """Prepare data with features"""
        print(f"Processing {len(df)} emails...")
        
        text_features = pd.DataFrame([
            self.extract_text_features(text) 
            for text in df[text_column]
        ])
        
        preprocessed_texts = [self.preprocess_text(text) for text in df[text_column]]
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(preprocessed_texts).toarray()
        
        X = np.hstack([text_features.values, tfidf_features])
        y = df[label_column].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Phishing emails: {sum(y)} | Legitimate emails: {len(y) - sum(y)}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """Train the model"""
        print("\n" + "="*60)
        print("TRAINING PHISHING DETECTOR")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=20, 
            random_state=self.random_state
        )
        
        self.model.fit(X_train, y_train)
        
        skf = StratifiedKFold(n_splits=5)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=skf, scoring='f1')
        
        print(f"\nCross-validation F1 Scores: {cv_scores}")
        print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.evaluate(X_test, y_test)
        
        return self.model, self.metrics
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
              target_names=['Legitimate', 'Phishing']))
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives (TN): {cm[0,0]}")
        print(f"  False Positives (FP): {cm[0,1]}")
        print(f"  False Negatives (FN): {cm[1,0]}")
        print(f"  True Positives (TP): {cm[1,1]}")
        
        f1 = f1_score(y_test, y_pred)
        print(f"\nF1-Score: {f1:.4f}")
        
        self.metrics = {
            'roc_auc': roc_auc,
            'f1_score': f1,
            'confusion_matrix': cm.tolist()
        }
    
    def predict_email(self, email_text):
        """Predict if single email is phishing"""
        text_features = pd.DataFrame([self.extract_text_features(email_text)])
        preprocessed = self.preprocess_text(email_text)
        tfidf_features = self.tfidf_vectorizer.transform([preprocessed]).toarray()
        X = np.hstack([text_features.values, tfidf_features])
        
        prediction = self.model.predict(X)[0]
        confidence = max(self.model.predict_proba(X)[0])
        
        return {
            'is_phishing': bool(prediction),
            'confidence': float(confidence)
        }
    
    def save_model(self, filepath):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'vectorizer': self.tfidf_vectorizer,
            'metrics': self.metrics
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")


if __name__ == "__main__":
    sample_data = {
        'email_body': [
            'Click here to verify your account immediately',
            'Meeting scheduled for tomorrow at 2pm',
            'Urgent: Update payment information now',
            'Project update: Q4 goals achieved',
            'Confirm your identity within 24 hours',
            'Thank you for using our service',
            'ACT NOW: Your account will be closed',
            'Weekly status report attached',
            'VERIFY ACCOUNT: Click link below',
            'Coffee meeting next week?',
        ] * 50,
        'is_phishing': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 50
    }
    
    df = pd.DataFrame(sample_data)
    
    detector = PhishingEmailDetector()
    X, y = detector.prepare_data(df)
    detector.train(X, y)
    
    detector.save_model('phishing_model.pkl')
    
    test_email = "Verify your account by clicking this suspicious link now"
    result = detector.predict_email(test_email)
    print(f"\n\nPrediction for test email:")
    print(f"Is Phishing: {result['is_phishing']}")
    print(f"Confidence: {result['confidence']:.2%}")
