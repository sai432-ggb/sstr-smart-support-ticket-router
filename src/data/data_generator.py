"""
Synthetic Support Ticket Data Generator
Generates realistic support ticket data for training and testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple
import json


class TicketDataGenerator:
    """Generate synthetic support ticket data"""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Define ticket categories and their characteristics
        self.categories = {
            "Technical Support": {
                "keywords": ["error", "bug", "crash", "not working", "broken", "issue", 
                            "problem", "fails", "unable", "cannot", "won't", "doesn't"],
                "subjects": [
                    "Application keeps crashing",
                    "Error 500 when trying to login",
                    "Cannot access my dashboard",
                    "Features not loading properly",
                    "System timeout error",
                    "API integration failing"
                ],
                "priority_dist": {"critical": 0.15, "high": 0.35, "medium": 0.35, "low": 0.15}
            },
            "Billing & Payments": {
                "keywords": ["payment", "invoice", "charge", "billing", "refund", "subscription",
                            "price", "cost", "paid", "credit card", "transaction"],
                "subjects": [
                    "Incorrect charge on my account",
                    "Need invoice for last month",
                    "Subscription renewal issue",
                    "Payment method not accepted",
                    "Request for refund",
                    "Double charged for service"
                ],
                "priority_dist": {"critical": 0.10, "high": 0.25, "medium": 0.45, "low": 0.20}
            },
            "Product Inquiry": {
                "keywords": ["features", "how to", "capabilities", "pricing", "plans",
                            "upgrade", "demo", "information", "details", "options"],
                "subjects": [
                    "What features are included in Enterprise plan?",
                    "How do I upgrade my account?",
                    "Pricing information for team accounts",
                    "Comparing different subscription tiers",
                    "Demo request for new features",
                    "Integration capabilities question"
                ],
                "priority_dist": {"critical": 0.05, "high": 0.15, "medium": 0.50, "low": 0.30}
            },
            "Account Management": {
                "keywords": ["account", "password", "username", "profile", "settings",
                            "reset", "change", "update", "delete", "access"],
                "subjects": [
                    "Cannot reset my password",
                    "Need to update account information",
                    "Change email address on account",
                    "Delete my account request",
                    "Security settings help",
                    "User access management"
                ],
                "priority_dist": {"critical": 0.08, "high": 0.27, "medium": 0.45, "low": 0.20}
            },
            "Feature Request": {
                "keywords": ["suggest", "feature", "enhancement", "improvement", "add",
                            "would like", "request", "propose", "idea", "could you"],
                "subjects": [
                    "Suggestion for new export feature",
                    "Request dark mode option",
                    "Add mobile app support",
                    "Integration with third-party tools",
                    "Bulk editing capabilities",
                    "Custom reporting features"
                ],
                "priority_dist": {"critical": 0.02, "high": 0.08, "medium": 0.40, "low": 0.50}
            },
            "Bug Report": {
                "keywords": ["bug", "glitch", "malfunction", "error", "incorrect", "wrong",
                            "broken", "not working", "unexpected", "weird behavior"],
                "subjects": [
                    "Data not syncing correctly",
                    "UI elements overlapping",
                    "Report generation producing wrong data",
                    "Search function not working",
                    "Email notifications not sending",
                    "Calendar integration bug"
                ],
                "priority_dist": {"critical": 0.20, "high": 0.40, "medium": 0.30, "low": 0.10}
            },
            "General Inquiry": {
                "keywords": ["question", "wondering", "curious", "help", "guidance",
                            "clarify", "explain", "understand", "confused", "general"],
                "subjects": [
                    "General question about service",
                    "Help understanding analytics",
                    "Clarification on terms of service",
                    "Best practices inquiry",
                    "Documentation clarification",
                    "Training resources availability"
                ],
                "priority_dist": {"critical": 0.03, "high": 0.12, "medium": 0.45, "low": 0.40}
            }
        }
        
        # Customer segments
        self.customer_tiers = ["free", "business", "enterprise"]
        self.tier_distribution = [0.60, 0.30, 0.10]
        
        # Email domains for realism
        self.email_domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com",
                             "business.com", "enterprise.org", "startup.io"]
    
    def generate_description(self, category: str, priority: str, subject: str) -> str:
        """Generate realistic ticket description"""
        
        templates = {
            "critical": [
                f"URGENT: {subject}. This is a critical issue affecting our entire team. "
                f"We need immediate assistance as production is down. This started {random.randint(1,4)} hours ago.",
                f"Emergency situation - {subject}. Multiple users are impacted and we're losing business. "
                f"Please escalate this immediately. Our operations are completely blocked.",
            ],
            "high": [
                f"We're experiencing {subject.lower()}. This is causing significant disruption "
                f"to our workflow. Can someone please help us resolve this today?",
                f"{subject}. This is affecting {random.randint(5,20)} users and needs attention soon. "
                f"We've tried basic troubleshooting but the issue persists.",
            ],
            "medium": [
                f"I'm having an issue with {subject.lower()}. It's not blocking us completely "
                f"but would appreciate help when possible.",
                f"{subject}. Looking for assistance with this. Not critical but would like "
                f"to resolve in the next few days if possible.",
            ],
            "low": [
                f"Quick question about {subject.lower()}. Not urgent but curious about the details.",
                f"{subject}. This is a minor issue. Would appreciate guidance when you have time.",
            ]
        }
        
        base_description = random.choice(templates[priority])
        
        # Add category-specific details
        keywords = self.categories[category]["keywords"]
        extra_detail = f" Specifically related to {random.choice(keywords)}."
        
        # Add customer context
        contexts = [
            " We've been using your service for {} months.".format(random.randint(1,36)),
            " This is the first time we've encountered this.",
            " Similar issues occurred last week.",
            " Our team relies heavily on this feature.",
            ""
        ]
        
        return base_description + extra_detail + random.choice(contexts)
    
    def generate_tickets(self, n_tickets: int = 1000,
                         start_date: datetime = None,
                         end_date: datetime = None) -> pd.DataFrame:
        """
        Generate synthetic ticket dataset
        
        Args:
            n_tickets: Number of tickets to generate
            start_date: Start date for ticket timestamps
            end_date: End date for ticket timestamps
            
        Returns:
            DataFrame with synthetic ticket data
        """
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=180)
        if end_date is None:
            end_date = datetime.now()
        
        tickets = []
        
        for i in range(n_tickets):
            # Select category
            category = random.choice(list(self.categories.keys()))
            cat_info = self.categories[category]
            
            # Select priority based on category distribution
            priority = random.choices(
                list(cat_info["priority_dist"].keys()),
                weights=list(cat_info["priority_dist"].values())
            )[0]
            
            # Generate subject
            subject = random.choice(cat_info["subjects"])
            
            # Generate description
            description = self.generate_description(category, priority, subject)
            
            # Generate customer info
            customer_tier = random.choices(
                self.customer_tiers,
                weights=self.tier_distribution
            )[0]
            
            customer_name = f"User{i+1000}"
            customer_email = f"{customer_name.lower()}@{random.choice(self.email_domains)}"
            
            # Generate timestamp (more tickets during business hours)
            timestamp = self.generate_timestamp(start_date, end_date)
            
            # Additional fields
            ticket = {
                "ticket_id": f"TKT-{i+1:06d}",
                "created_at": timestamp,
                "subject": subject,
                "description": description,
                "category": category,
                "priority": priority,
                "customer_name": customer_name,
                "customer_email": customer_email,
                "customer_tier": customer_tier,
                "status": random.choice(["open", "in_progress", "resolved", "closed"]),
                "channel": random.choice(["email", "web", "chat", "phone"]),
            }
            
            tickets.append(ticket)
        
        df = pd.DataFrame(tickets)
        
        # Add derived features
        df['text_length'] = df['description'].str.len()
        df['word_count'] = df['description'].str.split().str.len()
        df['has_urgent_keywords'] = df['description'].str.lower().str.contains(
            'urgent|emergency|critical|asap|immediately'
        ).astype(int)
        
        return df
    
    def generate_timestamp(self, start_date: datetime, end_date: datetime) -> datetime:
        """Generate realistic timestamp (more tickets during business hours)"""
        
        # Random date between start and end
        time_delta = end_date - start_date
        random_days = random.randint(0, time_delta.days)
        date = start_date + timedelta(days=random_days)
        
        # Business hours bias (9 AM - 6 PM, weekdays)
        is_business_hours = random.random() < 0.7
        
        if is_business_hours and date.weekday() < 5:  # Monday-Friday
            hour = random.randint(9, 17)
        else:
            hour = random.randint(0, 23)
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return date.replace(hour=hour, minute=minute, second=second)
    
    def generate_time_series_data(self, days: int = 180) -> pd.DataFrame:
        """
        Generate time series data for forecasting
        
        Args:
            days: Number of days of historical data
            
        Returns:
            DataFrame with daily ticket counts
        """
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Base trend with seasonality
        t = np.arange(len(dates))
        trend = 50 + 0.1 * t  # Slight upward trend
        
        # Weekly seasonality (higher on weekdays)
        weekly = 15 * np.sin(2 * np.pi * t / 7)
        
        # Monthly seasonality
        monthly = 10 * np.sin(2 * np.pi * t / 30)
        
        # Random noise
        noise = np.random.normal(0, 5, len(dates))
        
        # Combine components
        ticket_counts = trend + weekly + monthly + noise
        ticket_counts = np.maximum(ticket_counts, 0).astype(int)  # Ensure non-negative
        
        # Create DataFrame
        ts_df = pd.DataFrame({
            'date': dates,
            'ticket_count': ticket_counts
        })
        
        # Add category breakdowns
        for category in self.categories.keys():
            weight = random.uniform(0.05, 0.25)
            ts_df[f'{category}_count'] = (ts_df['ticket_count'] * weight).astype(int)
        
        return ts_df
    
    def save_data(self, df: pd.DataFrame, filepath: str, format: str = 'csv'):
        """Save generated data to file"""
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"✓ Saved {len(df)} records to {filepath}")


def main():
    """Main function to generate and save data"""
    
    print("🎲 Generating synthetic support ticket data...")
    
    generator = TicketDataGenerator(seed=42)
    
    # Generate training data (6 months of tickets)
    print("\n📊 Generating training dataset...")
    train_df = generator.generate_tickets(
        n_tickets=5000,
        start_date=datetime.now() - timedelta(days=180),
        end_date=datetime.now() - timedelta(days=30)
    )
    generator.save_data(train_df, 'data/raw/train_tickets.csv')
    
    # Generate recent data (last month)
    print("\n📊 Generating recent dataset...")
    recent_df = generator.generate_tickets(
        n_tickets=500,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    generator.save_data(recent_df, 'data/raw/recent_tickets.csv')
    
    # Generate time series data
    print("\n📈 Generating time series data...")
    ts_df = generator.generate_time_series_data(days=180)
    generator.save_data(ts_df, 'data/raw/ticket_time_series.csv')
    
    # Print summary statistics
    print("\n📋 Dataset Summary:")
    print(f"Total tickets: {len(train_df) + len(recent_df)}")
    print(f"\nCategory distribution:")
    print(train_df['category'].value_counts())
    print(f"\nPriority distribution:")
    print(train_df['priority'].value_counts())
    print(f"\nCustomer tier distribution:")
    print(train_df['customer_tier'].value_counts())
    
    print("\n✅ Data generation complete!")


if __name__ == "__main__":
    main()