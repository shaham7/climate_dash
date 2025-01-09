import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')  # Suppress convergence warnings

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2}

class TimeSeriesModels:
    def __init__(self, train_size=0.8, forecast_years=7):
        self.train_size = train_size
        self.forecast_years = forecast_years
    
    def fit_prophet(self, data):
        model = Prophet(yearly_seasonality=True)
        model.fit(data)
        future = model.make_future_dataframe(periods=self.forecast_years, freq='YE')
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']]
    
    def fit_arima_emission(self, data):
        # using arima model cofig derived from R auto.arima 
        model = ARIMA(data, order=(0,2,1))
        model_fit = model.fit()
        
        predictions = model_fit.forecast(steps=self.forecast_years)
        return np.concatenate([model_fit.fittedvalues, predictions])
    
    def fit_arima_temp(self, data):
        # using arima model cofig derived from R auto.arima 
        model = ARIMA(data, order=(3,1,0))
        model_fit = model.fit()
        
        predictions = model_fit.forecast(steps=self.forecast_years)
        return np.concatenate([model_fit.fittedvalues, predictions])

def generate_forecasts(data_path):
    print("Loading data...")
    data = pd.read_csv(data_path)
    world_data = data[data['Country'] == 'World'].copy()
    
    models = TimeSeriesModels(forecast_years=7)
    evaluator = ModelEvaluator()
    
    print("Processing emissions data...")
    # Process emissions data
    emissions_df = pd.DataFrame({
        'ds': pd.to_datetime(world_data['Year'], format='%Y'),
        'y': world_data['Emissions']
    }).dropna()
    
    # Get Prophet predictions
    prophet_emissions_df = models.fit_prophet(emissions_df)
    
    # Get ARIMA predictions
    arima_emissions = models.fit_arima_emission(emissions_df['y'].values)
    
    print("Processing temperature data...")
    # Process temperature data
    temp_df = pd.DataFrame({
        'ds': pd.to_datetime(world_data['Year'], format='%Y'),
        'y': world_data['Global_Temperature']
    }).dropna()
    
    # Get Prophet predictions
    prophet_temp_df = models.fit_prophet(temp_df)
    
    # Get ARIMA predictions
    arima_temp = models.fit_arima_temp(temp_df['y'].values)
    
    print("Calculating metrics...")
    # Calculate metrics using only historical data
    emissions_metrics = {
        'Prophet': evaluator.calculate_metrics(
            emissions_df['y'].values,
            prophet_emissions_df['yhat'].values[:len(emissions_df)]
        ),
        'ARIMA': evaluator.calculate_metrics(
            emissions_df['y'].values,
            arima_emissions[:len(emissions_df)]
        )
    }
    
    temp_metrics = {
        'Prophet': evaluator.calculate_metrics(
            temp_df['y'].values,
            prophet_temp_df['yhat'].values[:len(temp_df)]
        ),
        'ARIMA': evaluator.calculate_metrics(
            temp_df['y'].values,
            arima_temp[:len(temp_df)]
        )
    }
    
    print("Saving forecasts...")
    # Create forecasts dataframe
    forecasts_df = pd.DataFrame({
        'Date': prophet_emissions_df['ds'],
        'Prophet_Emissions': prophet_emissions_df['yhat'],
        'ARIMA_Emissions': arima_emissions,
        'Prophet_Temperature': prophet_temp_df['yhat'],
        'ARIMA_Temperature': arima_temp
    })
    
    # Save all files
    print("Saving files...")
    forecasts_df.to_csv('output/forecasts.csv', index=False)
    pd.DataFrame(emissions_metrics).T.to_csv('output/emissions_metrics.csv')
    pd.DataFrame(temp_metrics).T.to_csv('output/temperature_metrics.csv')
    print("Done! Files saved in output directory.")

if __name__ == "__main__":
    import os
    
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')
        
    generate_forecasts('./assets/processed_data.csv')