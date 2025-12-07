"""
Priority Detection Model
Detects ticket priority using sentiment analysis and keyword matching
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from textblob import TextBlob
import re


class PriorityDetector:
    """
    Detect ticket priority using multiple signals:
    - Sentiment analysis
    - Urgency keywords
    - Text patterns
    - Customer tier
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize priority detector"""
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'sentiment_threshold': -0.3,
            'urgency_keywords': {
                'critical': [
                    'urgent', 'emergency', 'critical', 'asap', 'immediately',
                    'production down', 'system down', 'not working', 'broken',
                    'losing money', 'losing business', 'severe', 'major issue'
                ],
                'high': [
                    'important', 'soon', 'problem', 'issue', 'error',
                    'cannot', "can't", 'unable', 'failing', 'multiple users',
                    'team blocked', 'need help'
                ],
                'medium': [
                    'question', 'help', 'how to', 'clarification',
                    'wondering', 'guidance', 'advice', 'could you'
                ],
                'low': [
                    'curious', 'suggest', 'feedback', 'enhancement',
                    'feature request', 'nice to have', 'when possible'
                ]
            },
            'tier_boost': {
                'enterprise': 1,
                'business': 0,
                'free': -1
            }
        }
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze text sentiment using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Sentiment polarity (-1 to 1)
        """
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def count_urgency_keywords(self, text: str) -> Dict[str, int]:
        """
        Count urgency keywords in text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with counts per priority level
        """
        text_lower = str(text).lower()
        
        counts = {}
        for priority, keywords in self.config['urgency_keywords'].items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            counts[priority] = count
        
        return counts
    
    def detect_patterns(self, text: str) -> Dict[str, bool]:
        """
        Detect specific patterns indicating priority
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with pattern flags
        """
        text_lower = str(text).lower()
        
        patterns = {
            'has_caps': bool(re.search(r'[A-Z]{3,}', str(text))),
            'has_exclamation': '!' in text,
            'has_multiple_exclamation': '!!' in text or '!!!' in text,
            'has_numbers': bool(re.search(r'\d+', text)),
            'mentions_time': bool(re.search(
                r'\d+\s*(hour|minute|day|week|month)s?', text_lower
            )),
            'mentions_money': bool(re.search(
                r'\$|dollar|payment|revenue|cost|price', text_lower
            )),
            'mentions_users': bool(re.search(
                r'\d+\s*users?|multiple users|team|everyone', text_lower
            ))
        }
        
        return patterns
    
    def calculate_priority_score(self, 
                                  text: str,
                                  customer_tier: str = 'free') -> float:
        """
        Calculate priority score (0-100)
        
        Args:
            text: Combined ticket text
            customer_tier: Customer tier (enterprise, business, free)
            
        Returns:
            Priority score
        """
        score = 50.0  # Base score
        
        # 1. Sentiment analysis (-20 to +20)
        sentiment = self.analyze_sentiment(text)
        if sentiment < self.config['sentiment_threshold']:
            score += 20  # Negative sentiment = higher priority
        else:
            score -= 10 * sentiment  # Positive sentiment = lower priority
        
        # 2. Urgency keywords
        keyword_counts = self.count_urgency_keywords(text)
        score += keyword_counts.get('critical', 0) * 15
        score += keyword_counts.get('high', 0) * 10
        score += keyword_counts.get('medium', 0) * 5
        score -= keyword_counts.get('low', 0) * 5
        
        # 3. Pattern detection
        patterns = self.detect_patterns(text)
        if patterns['has_caps']:
            score += 10
        if patterns['has_multiple_exclamation']:
            score += 15
        elif patterns['has_exclamation']:
            score += 5
        if patterns['mentions_time']:
            score += 10
        if patterns['mentions_money']:
            score += 8
        if patterns['mentions_users']:
            score += 12
        
        # 4. Customer tier adjustment
        tier_boost = self.config['tier_boost'].get(customer_tier, 0)
        score += tier_boost * 5
        
        # Clamp score to 0-100
        score = max(0, min(100, score))
        
        return score
    
    def predict_priority(self, 
                        subject: str,
                        description: str,
                        customer_tier: str = 'free') -> Dict[str, Any]:
        """
        Predict ticket priority
        
        Args:
            subject: Ticket subject
            description: Ticket description
            customer_tier: Customer tier
            
        Returns:
            Dictionary with priority prediction and details
        """
        # Combine text (subject weighted more)
        combined_text = f"{subject} {subject} {description}"
        
        # Calculate score
        score = self.calculate_priority_score(combined_text, customer_tier)
        
        # Map score to priority
        if score >= 75:
            priority = 'critical'
        elif score >= 60:
            priority = 'high'
        elif score >= 40:
            priority = 'medium'
        else:
            priority = 'low'
        
        # Get contributing factors
        sentiment = self.analyze_sentiment(combined_text)
        keyword_counts = self.count_urgency_keywords(combined_text)
        patterns = self.detect_patterns(combined_text)
        
        return {
            'priority': priority,
            'score': score,
            'sentiment': sentiment,
            'urgency_keywords': keyword_counts,
            'patterns': patterns,
            'customer_tier': customer_tier,
            'confidence': min(abs(score - 50) / 50, 1.0)  # Distance from midpoint
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict priorities for batch of tickets
        
        Args:
            df: DataFrame with tickets
            
        Returns:
            DataFrame with priority predictions
        """
        results = []
        
        for _, row in df.iterrows():
            pred = self.predict_priority(
                subject=row.get('subject', ''),
                description=row.get('description', ''),
                customer_tier=row.get('customer_tier', 'free')
            )
            results.append(pred)
        
        result_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), result_df], axis=1)
    
    def explain_priority(self, 
                        subject: str,
                        description: str,
                        customer_tier: str = 'free') -> str:
        """
        Generate human-readable explanation for priority
        
        Args:
            subject: Ticket subject
            description: Ticket description
            customer_tier: Customer tier
            
        Returns:
            Explanation string
        """
        pred = self.predict_priority(subject, description, customer_tier)
        
        explanations = []
        
        # Priority result
        explanations.append(
            f"Priority: {pred['priority'].upper()} (score: {pred['score']:.1f}/100)"
        )
        
        # Sentiment
        if pred['sentiment'] < -0.3:
            explanations.append("- Strong negative sentiment detected")
        elif pred['sentiment'] < 0:
            explanations.append("- Negative sentiment detected")
        
        # Keywords
        if pred['urgency_keywords']['critical'] > 0:
            explanations.append(
                f"- Found {pred['urgency_keywords']['critical']} critical urgency keywords"
            )
        if pred['urgency_keywords']['high'] > 0:
            explanations.append(
                f"- Found {pred['urgency_keywords']['high']} high urgency keywords"
            )
        
        # Patterns
        if pred['patterns']['has_multiple_exclamation']:
            explanations.append("- Multiple exclamation marks indicate urgency")
        if pred['patterns']['has_caps']:
            explanations.append("- Capital letters suggest emphasis")
        if pred['patterns']['mentions_users']:
            explanations.append("- Multiple users affected")
        if pred['patterns']['mentions_money']:
            explanations.append("- Financial impact mentioned")
        if pred['patterns']['mentions_time']:
            explanations.append("- Time-sensitive language detected")
        
        # Customer tier
        if customer_tier == 'enterprise':
            explanations.append("- Enterprise customer (priority boost)")
        
        return "\n".join(explanations)


def main():
    """Example usage"""
    
    detector = PriorityDetector()
    
    # Test cases
    test_cases = [
        {
            'subject': 'URGENT: Production system down!',
            'description': 'Our entire production system has been down for 2 hours. '
                          'This is affecting all users and we are losing revenue. '
                          'Need immediate assistance!',
            'customer_tier': 'enterprise'
        },
        {
            'subject': 'Question about feature',
            'description': 'I was wondering if you could help me understand '
                          'how to use the export feature when you have time.',
            'customer_tier': 'free'
        },
        {
            'subject': 'Login issue',
            'description': 'Cannot login to my account. Getting error 401. '
                          'This is blocking my team from working.',
            'customer_tier': 'business'
        }
    ]
    
    print("🎯 Priority Detection Examples:\n")
    print("=" * 70)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Subject: {case['subject']}")
        print(f"Customer: {case['customer_tier']}\n")
        
        explanation = detector.explain_priority(
            case['subject'],
            case['description'],
            case['customer_tier']
        )
        print(explanation)
        print("-" * 70)


if __name__ == "__main__":
    main()