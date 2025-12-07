"""
Ticket Demand Forecasting Model
Time series forecasting for resource planning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not available")

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


class TicketForecaster:
    """
    Time series forecasting for ticket volume prediction
    Uses Prophet for trend/seasonality or SARIMA as fallback
    """
    
    def __init__(self, method: str = 'prophet', config: Dict[str, Any] = None):
        """
        Initialize forecaster
        
        Args:
            method: 'prophet' or 'sarima'
            config: Configuration dictionary
        """
        self.method = method if (method == 'prophet' and PROPHET_AVAILABLE) else 'sarima'
        self.config = config or self._default_config()
        self.model = None
        self.history = None
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'prophet': {
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10
            },
            'sarima': {
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 7),
                'trend': 'c'
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, 
                     date_col: str = 'date',
                     value_col: str = 'ticket_count') -> pd.DataFrame:
        """
        Prepare data for forecasting
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            value_col: Value column name
            
        Returns:
            Prepared DataFrame
        """
        # Ensure datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Remove any duplicates
        df = df.drop_duplicates(subset=[date_col])
        
        # Fill missing dates with 0
        date_range = pd.date_range(
            start=df[date_col].min(),
            end=df[date_col].max(),
            freq='D'
        )
        full_df = pd.DataFrame({date_col: date_range})
        df = full_df.merge(df, on=date_col, how='left')
        df[value_col] = df[value_col].fillna(0)
        
        return df
    
    def fit_prophet(self, df: pd.DataFrame) -> None:
        """Fit Prophet model"""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available")
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = pd.DataFrame({
            'ds': df['date'],
            'y': df['ticket_count']
        })
        
        # Create and configure model
        self.model = Prophet(
            seasonality_mode=self.config['prophet']['seasonality_mode'],
            yearly_seasonality=self.config['prophet']['yearly_seasonality'],
            weekly_seasonality=self.config['prophet']['weekly_seasonality'],
            daily_seasonality=self.config['prophet']['daily_seasonality'],
            changepoint_prior_scale=self.config['prophet']['changepoint_prior_scale'],
            seasonality_prior_scale=self.config['prophet']['seasonality_prior_scale']
        )
        
        # Fit model
        self.model.fit(prophet_df)
    
    def fit_sarima(self, df: pd.DataFrame) -> None:
        """Fit SARIMA model"""
        # Prepare time series
        ts = df.set_index('date')['ticket_count']
        
        # Fit SARIMA
        self.model = SARIMAX(
            ts,
            order=self.config['sarima']['order'],
            seasonal_order=self.config['sarima']['seasonal_order'],
            trend=self.config['sarima']['trend']
        )
        
        self.model = self.model.fit(disp=False)
    
    def fit(self, df: pd.DataFrame) -> 'TicketForecaster':
        """
        Fit forecasting model
        
        Args:
            df: Historical data with 'date' and 'ticket_count' columns
            
        Returns:
            self
        """
        print(f"🔧 Training {self.method.upper()} forecaster...")
        
        # Prepare data
        df = self.prepare_data(df)
        self.history = df.copy()
        
        # Fit appropriate model
        if self.method == 'prophet':
            self.fit_prophet(df)
        else:
            self.fit_sarima(df)
        
        print("✓ Forecaster trained successfully")
        return self
    
    def predict_prophet(self, periods: int) -> pd.DataFrame:
        """Generate Prophet forecast"""
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq='D')
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Extract relevant columns
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        result = result.rename(columns={
            'ds': 'date',
            'yhat': 'forecast',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound'
        })
        
        # Ensure non-negative forecasts
        result['forecast'] = result['forecast'].clip(lower=0)
        result['lower_bound'] = result['lower_bound'].clip(lower=0)
        result['upper_bound'] = result['upper_bound'].clip(lower=0)
        
        return result
    
    def predict_sarima(self, periods: int) -> pd.DataFrame:
        """Generate SARIMA forecast"""
        # Generate forecast
        forecast_result = self.model.get_forecast(steps=periods)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
        
        # Create result DataFrame
        last_date = self.history['date'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        result = pd.DataFrame({
            'date': future_dates,
            'forecast': forecast.values,
            'lower_bound': conf_int.iloc[:, 0].values,
            'upper_bound': conf_int.iloc[:, 1].values
        })
        
        # Ensure non-negative forecasts
        result['forecast'] = result['forecast'].clip(lower=0)
        result['lower_bound'] = result['lower_bound'].clip(lower=0)
        result['upper_bound'] = result['upper_bound'].clip(lower=0)
        
        return result
    
    def forecast(self, periods: int = 7) -> pd.DataFrame:
        """
        Generate forecast
        
        Args:
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecast
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if self.method == 'prophet':
            return self.predict_prophet(periods)
        else:
            return self.predict_sarima(periods)
    
    def forecast_by_category(self, 
                            df: pd.DataFrame,
                            category_col: str = 'category',
                            periods: int = 7) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for each category
        
        Args:
            df: Historical data with category column
            category_col: Category column name
            periods: Number of periods to forecast
            
        Returns:
            Dictionary mapping categories to forecast DataFrames
        """
        forecasts = {}
        categories = df[category_col].unique()
        
        for category in categories:
            cat_df = df[df[category_col] == category].copy()
            
            # Aggregate by date
            cat_df = cat_df.groupby('date').size().reset_index(name='ticket_count')
            
            # Prepare data
            cat_df = self.prepare_data(cat_df)
            
            # Create and fit model
            forecaster = TicketForecaster(method=self.method, config=self.config)
            forecaster.fit(cat_df)
            
            # Generate forecast
            forecast = forecaster.forecast(periods)
            forecast['category'] = category
            
            forecasts[category] = forecast
        
        return forecasts
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate forecast accuracy
        
        Args:
            test_df: Test data with actual values
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Generate forecast for test period
        test_df = self.prepare_data(test_df)
        periods = len(test_df)
        forecast_df = self.forecast(periods)
        
        # Align dates
        merged = test_df.merge(forecast_df, on='date', how='inner')
        
        if len(merged) == 0:
            raise ValueError("No overlapping dates between test and forecast")
        
        y_true = merged['ticket_count'].values
        y_pred = merged['forecast'].values
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100  # +1 to avoid division by zero
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'n_samples': len(merged)
        }
        
        print(f"📊 Forecast Evaluation:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        return metrics
    
    def plot_forecast(self, forecast_df: pd.DataFrame, 
                     historical_df: pd.DataFrame = None):
        """
        Plot forecast with historical data (requires matplotlib)
        
        Args:
            forecast_df: Forecast DataFrame
            historical_df: Historical data (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            if historical_df is not None:
                plt.plot(historical_df['date'], historical_df['ticket_count'],
                        label='Historical', color='blue', linewidth=2)
            
            # Plot forecast
            plt.plot(forecast_df['date'], forecast_df['forecast'],
                    label='Forecast', color='red', linewidth=2, linestyle='--')
            
            # Plot confidence interval
            plt.fill_between(forecast_df['date'],
                           forecast_df['lower_bound'],
                           forecast_df['upper_bound'],
                           alpha=0.3, color='red', label='95% CI')
            
            plt.xlabel('Date')
            plt.ylabel('Ticket Count')
            plt.title('Ticket Volume Forecast')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("⚠️ Matplotlib not available for plotting")
    
    def save(self, filepath: str):
        """Save model"""
        model_data = {
            'model': self.model,
            'method': self.method,
            'config': self.config,
            'history': self.history
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Forecaster saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.method = model_data['method']
        self.config = model_data['config']
        self.history = model_data['history']
        print(f"✓ Forecaster loaded from {filepath}")


def main():
    """Example usage"""
    
    # Load time series data
    print("📂 Loading time series data...")
    df = pd.read_csv('data/raw/ticket_time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Split into train and test
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Train forecaster
    forecaster = TicketForecaster(method='sarima')  # Use SARIMA for wider compatibility
    forecaster.fit(train_df)
    
    # Generate 7-day forecast
    forecast = forecaster.forecast(periods=7)
    print("\n📈 7-Day Forecast:")
    print(forecast[['date', 'forecast', 'lower_bound', 'upper_bound']])
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    metrics = forecaster.evaluate(test_df)
    
    # Save model
    forecaster.save('models/saved_models/forecaster_v1.pkl')


if __name__ == "__main__":
    main()