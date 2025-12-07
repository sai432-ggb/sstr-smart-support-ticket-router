"""
Model Training Script
Trains all models (classifier, priority detector, forecaster)
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifier import TicketClassifier
from src.models.priority_detector import PriorityDetector
from src.models.forecaster import TicketForecaster


def ensure_directories():
    """Ensure required directories exist"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/features',
        'models/saved_models',
        'models/metrics',
        'models/experiments'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def train_classifier():
    """Train ticket classification model"""
    print("\n" + "="*70)
    print("🎯 TRAINING TICKET CLASSIFIER")
    print("="*70 + "\n")
    
    # Load data
    print("📂 Loading training data...")
    try:
        df = pd.read_csv('data/raw/train_tickets.csv')
        print(f"✓ Loaded {len(df)} tickets")
    except FileNotFoundError:
        print("❌ Training data not found!")
        print("Please run: python scripts/generate_data.py")
        return None
    
    # Print data distribution
    print("\n📊 Category distribution:")
    print(df['category'].value_counts())
    print("\n📊 Priority distribution:")
    print(df['priority'].value_counts())
    
    # Split data
    print("\n🔀 Splitting data (80/20)...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['category']
    )
    print(f"Training set: {len(train_df)} tickets")
    print(f"Test set: {len(test_df)} tickets")
    
    # Train model
    print("\n🔧 Training classifier...")
    classifier = TicketClassifier()
    classifier.fit(train_df, target_col='category')
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    results = classifier.evaluate(test_df, target_col='category')
    
    # Show feature importance
    print("\n🔍 Top 10 Most Important Features:")
    importance_df = classifier.get_feature_importance(top_n=10)
    print(importance_df.to_string(index=False))
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    test_cases = [
        {
            'subject': 'Production server down!',
            'description': 'Our main server crashed and all users are affected. Need immediate help!'
        },
        {
            'subject': 'Question about pricing',
            'description': 'I was wondering what features are included in the Business plan?'
        },
        {
            'subject': 'Cannot process payment',
            'description': 'Getting declined error when trying to update my credit card'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n  Test {i}:")
        print(f"  Subject: {case['subject']}")
        result = classifier.predict_single(case['subject'], case['description'])
        print(f"  → Category: {result['category']} (confidence: {result['confidence']:.2%})")
    
    # Save model
    print("\n💾 Saving classifier...")
    os.makedirs('models/saved_models', exist_ok=True)
    classifier.save(
        'models/saved_models/classifier_v1.pkl',
        'models/saved_models/vectorizer_v1.pkl'
    )
    
    # Save metrics
    import json
    with open('models/metrics/classifier_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Metrics saved to models/metrics/classifier_metrics.json")
    
    print("\n✅ Classifier training complete!")
    return classifier


def train_forecaster():
    """Train demand forecasting model"""
    print("\n" + "="*70)
    print("📈 TRAINING DEMAND FORECASTER")
    print("="*70 + "\n")
    
    # Load time series data
    print("📂 Loading time series data...")
    try:
        df = pd.read_csv('data/raw/ticket_time_series.csv')
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Loaded {len(df)} days of data")
    except FileNotFoundError:
        print("❌ Time series data not found!")
        print("Please run: python scripts/generate_data.py")
        return None
    
    # Split data
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size].copy()
    test_df = df[train_size:].copy()
    
    print(f"Training period: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    
    # Train forecaster (use SARIMA for better compatibility)
    print("\n🔧 Training SARIMA forecaster...")
    forecaster = TicketForecaster(method='sarima')
    forecaster.fit(train_df[['date', 'ticket_count']])
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    try:
        metrics = forecaster.evaluate(test_df[['date', 'ticket_count']])
    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        metrics = {}
    
    # Generate sample forecast
    print("\n🔮 Generating 7-day forecast:")
    forecast = forecaster.forecast(periods=7)
    print(forecast[['date', 'forecast', 'lower_bound', 'upper_bound']].to_string(index=False))
    
    # Save model
    print("\n💾 Saving forecaster...")
    forecaster.save('models/saved_models/forecaster_v1.pkl')
    
    # Save metrics
    if metrics:
        import json
        with open('models/metrics/forecaster_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print("✓ Metrics saved to models/metrics/forecaster_metrics.json")
    
    print("\n✅ Forecaster training complete!")
    return forecaster


def test_priority_detector():
    """Test priority detection (rule-based, no training needed)"""
    print("\n" + "="*70)
    print("🎯 TESTING PRIORITY DETECTOR")
    print("="*70 + "\n")
    
    detector = PriorityDetector()
    
    test_cases = [
        {
            'subject': 'URGENT: System completely down!',
            'description': 'Production is down for 3 hours! All users affected! Need immediate help!',
            'tier': 'enterprise'
        },
        {
            'subject': 'Login issue',
            'description': 'Having trouble logging in. Can you help when you get a chance?',
            'tier': 'free'
        },
        {
            'subject': 'Feature suggestion',
            'description': 'It would be nice to have a dark mode option in the future.',
            'tier': 'business'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}:")
        print(f"Subject: {case['subject']}")
        print(f"Tier: {case['tier']}")
        print()
        
        explanation = detector.explain_priority(
            case['subject'],
            case['description'],
            case['tier']
        )
        print(explanation)
        print("-" * 70)
    
    print("\n✅ Priority detector test complete!")
    return detector


def main():
    """Main training pipeline"""
    print("\n" + "🚀 " + "="*66)
    print("🚀  SMART SUPPORT TICKET ROUTER - MODEL TRAINING PIPELINE")
    print("🚀 " + "="*66 + "\n")
    
    # Ensure directories exist
    ensure_directories()
    
    # Train models
    try:
        classifier = train_classifier()
        forecaster = train_forecaster()
        detector = test_priority_detector()
        
        print("\n" + "="*70)
        print("🎉 ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*70)
        print("\n📋 Summary:")
        print(f"  ✓ Classifier: {'Trained' if classifier else 'Failed'}")
        print(f"  ✓ Forecaster: {'Trained' if forecaster else 'Failed'}")
        print(f"  ✓ Priority Detector: {'Tested' if detector else 'Failed'}")
        
        print("\n🚀 Next Steps:")
        print("  1. Start the API: uvicorn src.api.main:app --reload")
        print("  2. Open API docs: http://localhost:8000/docs")
        print("  3. Test endpoints with sample tickets")
        
        print("\n💡 Pro Tip:")
        print("  Monitor model performance and retrain periodically")
        print("  Use: python scripts/evaluate_models.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()