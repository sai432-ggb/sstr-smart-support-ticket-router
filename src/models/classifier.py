"""
Support Ticket Classification Model
Multi-class classifier using ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available, using fallback models")


class TicketClassifier:
    """
    Multi-class ticket classification using ensemble methods
    Combines TF-IDF features with ensemble of classifiers
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the classifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.vectorizer = None
        self.label_encoder = None
        self.model = None
        self.feature_names = None
        self.classes_ = None
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 5,
            'max_df': 0.8,
            'random_state': 42,
            'ensemble_weights': {
                'random_forest': 0.4,
                'logistic': 0.3,
                'xgboost': 0.3
            }
        }
    
    def _create_ensemble_model(self) -> VotingClassifier:
        """Create ensemble model with multiple classifiers"""
        
        estimators = []
        weights = []
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.config['random_state'],
            n_jobs=-1,
            class_weight='balanced'
        )
        estimators.append(('rf', rf))
        weights.append(self.config['ensemble_weights']['random_forest'])
        
        # Logistic Regression
        lr = LogisticRegression(
            max_iter=1000,
            random_state=self.config['random_state'],
            class_weight='balanced',
            solver='saga',
            n_jobs=-1
        )
        estimators.append(('lr', lr))
        weights.append(self.config['ensemble_weights']['logistic'])
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=self.config['random_state'],
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            estimators.append(('xgb', xgb))
            weights.append(self.config['ensemble_weights']['xgboost'])
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        
        return ensemble
    
    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        import re
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Extract TF-IDF features from text
        
        Args:
            df: DataFrame with 'subject' and 'description' columns
            fit: Whether to fit the vectorizer
            
        Returns:
            Feature matrix
        """
        # Combine subject and description
        texts = (df['subject'] + ' ' + df['description']).apply(self.preprocess_text)
        
        if fit:
            # Create and fit vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.config['max_features'],
                ngram_range=self.config['ngram_range'],
                min_df=self.config['min_df'],
                max_df=self.config['max_df'],
                stop_words='english',
                sublinear_tf=True,
                strip_accents='unicode'
            )
            X = self.vectorizer.fit_transform(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
        else:
            if self.vectorizer is None:
                raise ValueError("Vectorizer not fitted. Call fit() first.")
            X = self.vectorizer.transform(texts)
        
        return X
    
    def fit(self, df: pd.DataFrame, target_col: str = 'category') -> 'TicketClassifier':
        """
        Train the classifier
        
        Args:
            df: Training DataFrame
            target_col: Name of target column
            
        Returns:
            self
        """
        print("🔧 Training Ticket Classifier...")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df[target_col])
        self.classes_ = self.label_encoder.classes_
        
        # Extract features
        print("📝 Extracting features...")
        X = self.extract_features(df, fit=True)
        
        # Create and train model
        print("🎯 Training ensemble model...")
        self.model = self._create_ensemble_model()
        self.model.fit(X, y)
        
        # Calculate training accuracy
        y_pred = self.model.predict(X)
        train_acc = accuracy_score(y, y_pred)
        print(f"✓ Training accuracy: {train_acc:.4f}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict categories
        
        Args:
            df: DataFrame with tickets
            
        Returns:
            Predicted categories
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X = self.extract_features(df, fit=False)
        y_pred = self.model.predict(X)
        
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            df: DataFrame with tickets
            
        Returns:
            Probability matrix
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X = self.extract_features(df, fit=False)
        return self.model.predict_proba(X)
    
    def predict_single(self, subject: str, description: str) -> Dict[str, Any]:
        """
        Predict category for a single ticket
        
        Args:
            subject: Ticket subject
            description: Ticket description
            
        Returns:
            Prediction dictionary with category, confidence, and probabilities
        """
        df = pd.DataFrame([{
            'subject': subject,
            'description': description
        }])
        
        category = self.predict(df)[0]
        proba = self.predict_proba(df)[0]
        confidence = float(np.max(proba))
        
        # Get top 3 predictions
        top_indices = np.argsort(proba)[-3:][::-1]
        top_predictions = [
            {
                'category': self.classes_[idx],
                'probability': float(proba[idx])
            }
            for idx in top_indices
        ]
        
        return {
            'category': category,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_probabilities': {
                cat: float(prob) 
                for cat, prob in zip(self.classes_, proba)
            }
        }
    
    def evaluate(self, df: pd.DataFrame, target_col: str = 'category') -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            df: Test DataFrame
            target_col: Name of target column
            
        Returns:
            Evaluation metrics
        """
        print("📊 Evaluating model...")
        
        y_true = df[target_col].values
        y_pred = self.predict(df)
        y_proba = self.predict_proba(df)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        class_metrics = {}
        for cls in self.classes_:
            if cls in report:
                class_metrics[cls] = {
                    'precision': report[cls]['precision'],
                    'recall': report[cls]['recall'],
                    'f1-score': report[cls]['f1-score'],
                    'support': report[cls]['support']
                }
        
        results = {
            'accuracy': accuracy,
            'macro_avg': {
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1-score': report['macro avg']['f1-score']
            },
            'weighted_avg': {
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1-score': report['weighted avg']['f1-score']
            },
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist()
        }
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
        print(f"{'='*50}\n")
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top important features
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not trained")
        
        # Get feature importance from Random Forest
        rf_model = self.model.named_estimators_['rf']
        importances = rf_model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save(self, model_path: str, vectorizer_path: str):
        """Save model and vectorizer"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, model_path.replace('.pkl', '_encoder.pkl'))
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Vectorizer saved to {vectorizer_path}")
    
    def load(self, model_path: str, vectorizer_path: str):
        """Load model and vectorizer"""
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(model_path.replace('.pkl', '_encoder.pkl'))
        self.classes_ = self.label_encoder.classes_
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"✓ Model loaded from {model_path}")


def main():
    """Example usage"""
    
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv('data/raw/train_tickets.csv')
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                          stratify=df['category'])
    
    # Train classifier
    classifier = TicketClassifier()
    classifier.fit(train_df)
    
    # Evaluate
    results = classifier.evaluate(test_df)
    
    # Test single prediction
    print("\n🔮 Testing single prediction:")
    result = classifier.predict_single(
        subject="Cannot login to account",
        description="I'm getting error 401 when trying to login. This is urgent!"
    )
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    # Save model
    classifier.save(
        'models/saved_models/classifier_v1.pkl',
        'models/saved_models/vectorizer_v1.pkl'
    )


if __name__ == "__main__":
    main()