"""
Generate Synthetic Training Data
Creates realistic support ticket data for model training
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_generator import TicketDataGenerator
from datetime import datetime, timedelta


def main():
    """Generate all required training data"""
    
    print("\n" + "="*70)
    print("🎲 GENERATING SYNTHETIC TRAINING DATA")
    print("="*70 + "\n")
    
    # Ensure directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data/features', exist_ok=True)
    
    # Initialize generator
    generator = TicketDataGenerator(seed=42)
    
    # Generate training dataset (6 months, 5000 tickets)
    print("📊 Generating main training dataset...")
    print("  • Period: Last 6 months")
    print("  • Size: 5,000 tickets")
    
    train_df = generator.generate_tickets(
        n_tickets=5000,
        start_date=datetime.now() - timedelta(days=180),
        end_date=datetime.now() - timedelta(days=30)
    )
    
    generator.save_data(train_df, 'data/raw/train_tickets.csv')
    
    # Generate validation dataset (last month, 1000 tickets)
    print("\n📊 Generating validation dataset...")
    print("  • Period: Last month")
    print("  • Size: 1,000 tickets")
    
    val_df = generator.generate_tickets(
        n_tickets=1000,
        start_date=datetime.now() - timedelta(days=30),
        end_date=datetime.now()
    )
    
    generator.save_data(val_df, 'data/raw/val_tickets.csv')
    
    # Generate recent test data (last week, 200 tickets)
    print("\n📊 Generating test dataset...")
    print("  • Period: Last week")
    print("  • Size: 200 tickets")
    
    test_df = generator.generate_tickets(
        n_tickets=200,
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now()
    )
    
    generator.save_data(test_df, 'data/raw/test_tickets.csv')
    
    # Generate time series data for forecasting
    print("\n📈 Generating time series data...")
    print("  • Period: Last 6 months (daily)")
    
    ts_df = generator.generate_time_series_data(days=180)
    generator.save_data(ts_df, 'data/raw/ticket_time_series.csv')
    
    # Generate sample data for demos
    print("\n📄 Generating sample data for demos...")
    
    sample_df = generator.generate_tickets(
        n_tickets=20,
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now()
    )
    
    generator.save_data(sample_df, 'data/raw/sample_tickets.csv')
    generator.save_data(sample_df, 'data/raw/sample_tickets.json', format='json')
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("📊 DATA GENERATION SUMMARY")
    print("="*70)
    
    all_tickets = len(train_df) + len(val_df) + len(test_df)
    
    print(f"\n📈 Total Tickets Generated: {all_tickets:,}")
    print(f"  • Training Set: {len(train_df):,} tickets")
    print(f"  • Validation Set: {len(val_df):,} tickets")
    print(f"  • Test Set: {len(test_df):,} tickets")
    
    print(f"\n📅 Time Series Data: {len(ts_df)} days")
    
    print("\n📋 Category Distribution (Training Set):")
    category_dist = train_df['category'].value_counts()
    for category, count in category_dist.items():
        pct = (count / len(train_df)) * 100
        print(f"  • {category:25s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n🎯 Priority Distribution (Training Set):")
    priority_dist = train_df['priority'].value_counts()
    for priority, count in priority_dist.items():
        pct = (count / len(train_df)) * 100
        print(f"  • {str(priority).capitalize():10s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n👥 Customer Tier Distribution (Training Set):")
    tier_dist = train_df['customer_tier'].value_counts()
    for tier, count in tier_dist.items():
        pct = (count / len(train_df)) * 100
        print(f"  • {str(priority).capitalize():10s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n📊 Channel Distribution (Training Set):")
    channel_dist = train_df['channel'].value_counts()
    for channel, count in channel_dist.items():
        pct = (count / len(train_df)) * 100
        print(f"  • {str(channel).capitalize():10s}: {count:4d} ({pct:5.1f}%)")
    
    # Data quality checks
    print("\n✅ Data Quality Checks:")
    print(f"  • No missing subjects: {train_df['subject'].notna().all()}")
    print(f"  • No missing descriptions: {train_df['description'].notna().all()}")
    print(f"  • Valid categories: {train_df['category'].notna().all()}")
    print(f"  • Valid priorities: {train_df['priority'].notna().all()}")
    print(f"  • Average text length: {train_df['text_length'].mean():.0f} characters")
    print(f"  • Average word count: {train_df['word_count'].mean():.0f} words")
    
    # Save metadata
    metadata = {
        'generation_date': datetime.now().isoformat(),
        'total_tickets': all_tickets,
        'training_tickets': len(train_df),
        'validation_tickets': len(val_df),
        'test_tickets': len(test_df),
        'time_series_days': len(ts_df),
        'categories': category_dist.to_dict(),
        'priorities': priority_dist.to_dict(),
        'customer_tiers': tier_dist.to_dict()
    }
    
    import json
    with open('data/raw/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n💾 Files Created:")
    print("  ✓ data/raw/train_tickets.csv")
    print("  ✓ data/raw/val_tickets.csv")
    print("  ✓ data/raw/test_tickets.csv")
    print("  ✓ data/raw/ticket_time_series.csv")
    print("  ✓ data/raw/sample_tickets.csv")
    print("  ✓ data/raw/sample_tickets.json")
    print("  ✓ data/raw/metadata.json")
    
    print("\n" + "="*70)
    print("✅ DATA GENERATION COMPLETE!")
    print("="*70)
    
    print("\n🚀 Next Steps:")
    print("  1. Train models: python scripts/train_models.py")
    print("  2. Or use Make: make train")
    print("  3. Start API: make api")
    
    print("\n💡 Pro Tips:")
    print("  • Adjust generator parameters in src/data/data_generator.py")
    print("  • Add custom categories and keywords for your use case")
    print("  • Increase ticket count for better model performance")
    print("  • Use real historical data if available")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())