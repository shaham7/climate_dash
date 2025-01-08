import dash
import pandas as pd 
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        # Calculate RMSE, MAE, MAPE, R² for checking model performance

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        return { 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2 }
    
class TimeSeriesModels:
    # class for ARIMA and Prophet
    
    def __init__(self, train_size=0.8, forecast_years=7): 
        self.train_size = train_size
        self.forecast_years = forecast_years
        self.scaler = MinMaxScaler()
    
    # Preparing data for modeling, splitting into train/test sets
    def prepare_data(self, data):
        train_size = int(len(data) * self.train_size)
        train = data[:train_size]
        test = data[train_size:]
        return train, test
    
    # Fit and predict using Prophet
    def fit_prophet(self, data):
        model = Prophet(yearly_seasonality=True)
        model.fit(data)

        future = model.make_future_dataframe(periods=self.forecast_years, freq='YE')
        forecast = model.predict(future)
        return forecast['yhat'].values, forecast['ds']
    
    # Fit and predict using ARIMA
    def fit_arima(self, data):
        model = ARIMA(data, order=(1,1,1))
        model_fit = model.fit()

        predictions = model_fit.forecast(steps=self.forecast_years)
        return np.concatenate([model_fit.fittedvalues, predictions])

class ClimateDashboard:
    def __init__(self, data):
        self.app = dash.Dash(__name__)
        self.data = data
        self.metrics = ['Emissions', 'GDP_per_capita', 'Renewable_Share', 'Energy_per_capita', 'Population', 'Emissions_per_GDP', 'Emissions_per_capita']
        # self.correlation_metrics = ['Emissions_per_capita', 'GDP_per_capita', 'Energy_per_capita', 'Renewable_Share', 'Population']

        self.time_series_models = TimeSeriesModels(forecast_years=7)
        self.model_evaluator = ModelEvaluator()
        
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                <title>Climate Dashboard</title>
                <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">
                {%metas%}
                {%favicon%}
                {%css%}
                <style>
                    body {
                        font-family: 'Open Sans', sans-serif;
                        background-color: #f0f5f9;
                        color: #2c3e50;
                        text-align: center;
                    }
                    h1, h2 {
                        color: #34495e;
                    }
                    .dash-dropdown {
                        font-family: 'Open Sans', sans-serif;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                {%config%}
                {%scripts%}
                {%renderer%}
            </body>
        </html>
        '''
        
        self.setup_layout()
        self.setup_callbacks()



    def setup_layout(self):
        years = sorted(self.data['Year'].unique())
        dropdown_style = {
            'width': '100%',
            'padding': '3px',
            'borderRadius': '3px',
            'cursor': 'pointer',
        }

        self.app.layout = html.Div([
            html.H1("Exploring Climate Trends and Forecasting Models"),

            # Intro Section
            html.Div([
                html.Div([
                    html.P(
                        "I set out to create this Dashboard to better understand climate change data and time series models like ARIMA and Prophet. "
                        "I used data from ten countries and the world, spanning metrics such as greenhouse gas (GHG) emissions, GDP per capita, renewable energy consumption, and global temperatures, with data collected from 1970 to 2023."
                    ),
                    html.Ul([
                        html.Li("To deepen my understanding of building dashboards with Python and Dash."),
                        html.Li("To explore and apply time series forecasting models, such as ARIMA and Prophet."),
                        html.Li("To provide insightful visualizations and comparisons of climate-related metrics across countries and over time."),
                    ]),
                    html.P(
                        "This dashboard includes sections for time series analysis, comparative metric visualizations, details about forecasting models and model perfomance comparisons."
                    ),
                    html.P(
                        "Data sources: Our World in Data, International Monetary Fund, The World Bank, and The Emissions Database for Global Atmospheric Research report.",
                        style={'fontStyle': 'italic', 'fontSize': 'smaller'}
                    )
                ], style={'textAlign': 'left', 'padding': '20px', 'margin': '20px'})
            ], style={ 'padding': '20px', 'margin': '20px', 'textAlign': 'left'}),   


            # Time Series Section
            html.Div([
                html.H2("Time Series Plot"),
                html.Div([
                    html.Div([
                        html.Label("Select Country"),
                        dcc.Dropdown(
                            id='country-selector',
                            options=[{'label': c, 'value': c} for c in sorted(self.data['Country'].unique())],
                            value='United States',
                            style=dropdown_style, 
                            searchable= False
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'padding':'20px'}),
                    
                    html.Div([
                        html.Label("Select Metric"),
                        dcc.Dropdown(
                            id='metric-selector',
                            options=[{'label': m.replace('_', ' '), 'value': m} for m in self.metrics],
                            value='Emissions',
                            style=dropdown_style, 
                            searchable= False
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'margin': '20px'}),
                ], style={'padding': '20px'}),
                
                dcc.Graph(id='time-series-plot'),
            ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'margin': '20px', 'borderRadius': '10px'}),
            
            # # Correlation Analysis Section
            # need to learn more about correlation to implimenet this 
            
            # Comparative Analysis of Metrics by Country
            html.Div([
                html.H2("Comparative Analysis of Metrics by Country"),
                html.Div([
                    html.Div([
                        html.Label("Select Metric"),
                        dcc.Dropdown(
                            id='ranking-metric-selector',
                            options=[{'label': m.replace('_', ' '), 'value': m} for m in self.metrics],
                            value='Emissions_per_capita',
                            style=dropdown_style, 
                            searchable= False
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'padding':'20px'}),
                    
                    html.Div([
                        html.Label("Select Year"),
                        dcc.Dropdown(
                            id='ranking-year-selector',
                            options=[{'label': str(y), 'value': y} for y in years],
                            value=max(years),
                            style=dropdown_style, 
                            searchable= False
                        )
                    ], style={'width': '30%', 'display': 'inline-block'}),
                ], style={'padding': '20px'}),
                
                dcc.Graph(id='ranking-plot'),
            ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'margin': '20px', 'borderRadius': '10px'}),

            # Explanation Section
            html.Div([
                html.H3("Prophet and ARIMA Models"),
                html.Div([
                    html.H4("Prophet Model:"),
                    html.P(
                        """
                            Prophet, developed by Facebook, is a user-friendly forecasting tool designed for time series with clear trends and seasonality, even if there are missing values or outliers. It decomposes data into three components: long-term trends, seasonal patterns (like yearly or weekly cycles), and the effects of specific events (holidays). 
                            This makes it particularly effective for time series with strong seasonality and trend changes. Prophet is popular in applications like sales, traffic, and climate forecasting, as well as in various other fields like economics, finance, and social sciences, where it excels at handling complex patterns with minimal tuning.
                        """
                    ),
                    html.H4("ARIMA Model:"),
                    html.P(
                        """
                            ARIMA (AutoRegressive Integrated Moving Average) is a statistical model for forecasting stationary time series. It breaks data into three parts: AR (uses past values to predict future values), I (makes data stationary through differencing), and MA (uses past errors to refine predictions). 
                            Suitable for relatively stable and linear trends, ARIMA can handle seasonality with extensions like SARIMA (Seasonal ARIMA), making it effective for many forecasting scenarios. 
                            However, ARIMA models can be more challenging to tune than Prophet models, as they require careful selection of the model order (p, d, q) and potentially seasonal order (P, D, Q).
                        """
                    ),
                    
                ], style={'textAlign': 'left', 'padding': '20px', 'margin': '20px'})
            ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'margin': '20px', 'borderRadius': '10px'}),   
            
            # World Forecast Section
            html.Div([
                html.H2("World Forecasts"),
                
                # Forecast plots
                dcc.Graph(id='world-forecast'),
                
            ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'margin': '20px', 'borderRadius': '10px'}),

            # Model Comparison
            html.Div([
                html.H2("Model Performance Comparison"),
                
                # Model Comparison Section
                html.Div([
                    # html.H3("Model Performance Comparison"),
                    html.Div([
                        html.Div([
                            html.H4("Emissions Forecast Metrics"),
                            html.Div(id='emissions-metrics-table')
                        ], style={'width': '45%', 'display': 'inline-block', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'padding': '10px', 'margin': '10px'}),
                        
                        html.Div([
                            html.H4("Temperature Forecast Metrics"),
                            html.Div(id='temperature-metrics-table')
                        ], style={'width': '45%', 'display': 'inline-block', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'padding': '10px', 'margin': '10px'})
                    ])
                ]),

            html.Div([
                    html.H4("Key Metrics:"),
                    html.Ul([
                        html.Li( [html.B("RMSE (Root Mean Square Error): "), "Measures the average error magnitude; lower values indicate better performance."]),
                        html.Li( [html.B("MAE (Mean Absolute Error): "), "Represents the average absolute error; less sensitive to large deviations compared to RMSE."]),
                        html.Li( [html.B("MAPE (Mean Absolute Percentage Error): "), "Indicates the average percentage error; values below 10% are ideal."]),
                        html.Li( [html.B("R² (Coefficient of Determination): "), "Explains the variability in the data; closer to 1 signifies a better fit."]),
                    ]),
                    html.H4("Performance Summary:"),
                    html.Ul([
                        html.Li( [html.B("Emissions Forecast: "), "Prophet significantly outperforms ARIMA for emissions forecasting, with much lower errors and a near-perfect R², making it the preferred model for this metric."]),
                        html.Li( [html.B("Temperature Forecast: "), "Both models show high relative errors for temperature predictions. While Prophet has a slight edge in RMSE and R², neither model is ideal for this metric without further refinement."]),
                    ]),
                    html.H4("Model Suitability:"),
                    html.Ul([
                        html.Li( [html.B("Best for Emissions:"), "Prophet, with excellent accuracy and a strong fit."]),
                        html.Li( [html.B("For Temperature: "), "Neither model performs exceptionally."]),
                    ]),
                ], style={'textAlign': 'left', 'padding': '20px', 'margin': '20px'})
            ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'margin': '20px', 'borderRadius': '10px'})


        ], style={'maxWidth': '1200px', 'margin': 'auto', 'padding': '20px'})
    
    def setup_callbacks(self):
        @self.app.callback(
            Output('time-series-plot', 'figure'),
            [Input('country-selector', 'value'), Input('metric-selector', 'value')]
        )
        def update_time_series(country, metric):
            df = self.data[self.data['Country'] == country]
            fig = px.line(df, x='Year', y=metric,
                         title=f'{metric.replace("_", " ")} Over Time - {country}')
            return fig

        @self.app.callback(
            Output('ranking-plot', 'figure'),
            [Input('ranking-metric-selector', 'value'), Input('ranking-year-selector', 'value')]
        )
        def update_ranking(metric, year):
            df = self.data[self.data['Year'] == year]
            df = df[df['Country'] != 'World']
            
            # Get top and bottom 5
            top5 = df.nlargest(5, metric)
            bottom5 = df.nsmallest(5, metric)
            plot_data = pd.concat([top5, bottom5])
            
            fig = px.bar(plot_data, 
                        x='Country', y=metric,
                        title=f'Top and Bottom 5 Countries by {metric.replace("_", " ")} ({year})',
                        color=metric,
                        color_continuous_scale='RdYlBu_r')
            
            return fig
        
        @self.app.callback(
            [Output('world-forecast', 'figure'), Output('emissions-metrics-table', 'children'), Output('temperature-metrics-table', 'children')],
            [Input('metric-selector', 'value')]
        )
        def update_forecast_and_metrics(_):
            world_data = self.data[self.data['Country'] == 'World'].copy()
            
            # Create future dates for plotting
            last_year = world_data['Year'].max()
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=('World Emissions Forecast Comparison', 'Global Temperature Forecast Comparison'))
            
            # Process emissions data
            emissions_df = pd.DataFrame({
                'ds': pd.to_datetime(world_data['Year'], format='%Y'),
                'y': world_data['Emissions']
            }).dropna()
            
            # Get predictions from different models
            prophet_emissions, prophet_dates = self.time_series_models.fit_prophet(emissions_df)
            arima_emissions = self.time_series_models.fit_arima(emissions_df['y'].values)
            
            # Create future dates array for plotting
            historical_dates = emissions_df['ds']
            future_dates = pd.date_range(start=historical_dates.iloc[-1], periods=7, freq='YE')[1:]
            all_dates = pd.concat([historical_dates, pd.Series(future_dates)])
            
            # Calculate metrics (using only historical data)
            emissions_metrics = {
                'Prophet': self.model_evaluator.calculate_metrics(emissions_df['y'].values, prophet_emissions[:len(emissions_df)]),
                'ARIMA': self.model_evaluator.calculate_metrics(emissions_df['y'].values, arima_emissions[:len(emissions_df)])
            }
            
            # Plot emissions with extended forecast
            fig.add_trace(go.Scatter(x=historical_dates, y=emissions_df['y'], name='Historical Emissions', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=all_dates, y=prophet_emissions, name='Prophet Forecast', line=dict(color='red', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=all_dates, y=arima_emissions, name='ARIMA Forecast', line=dict(color='green', dash='dash')), row=1, col=1)
            
            # Add vertical line at present
            fig.add_vline(x=str(last_year), line_dash="dash", line_color="gray", row=1, col=1)
            
            # Similar process for temperature data
            temp_df = pd.DataFrame({
                'ds': pd.to_datetime(world_data['Year'], format='%Y'),
                'y': world_data['Global_Temperature']
            }).dropna()
            
            prophet_temp, _ = self.time_series_models.fit_prophet(temp_df)
            arima_temp = self.time_series_models.fit_arima(temp_df['y'].values)
            
            # Calculate temperature metrics
            temp_metrics = {
                'Prophet': self.model_evaluator.calculate_metrics(temp_df['y'].values, prophet_temp[:len(temp_df)]),
                'ARIMA': self.model_evaluator.calculate_metrics(temp_df['y'].values, arima_temp[:len(temp_df)]),
            }
            
            # Plot temperature forecasts
            historical_dates_temp = temp_df['ds']
            all_dates_temp = pd.concat([historical_dates_temp, pd.Series(future_dates)])
            
            fig.add_trace(go.Scatter(x=historical_dates_temp, y=temp_df['y'], name='Historical Temperature', line=dict(color='orange')), row=2, col=1)
            fig.add_trace(go.Scatter(x=all_dates_temp, y=prophet_temp, name='Prophet Forecast', line=dict(color='red', dash='dash')), row=2, col=1)
            fig.add_trace(go.Scatter(x=all_dates_temp, y=arima_temp, name='ARIMA Forecast', line=dict(color='green', dash='dash')), row=2, col=1)
            
            # Add vertical line at present
            fig.add_vline(x=str(last_year), line_dash="dash", line_color="gray", row=2, col=1)
            
            fig.update_layout(height=1000, showlegend=True)
            
            # Create metrics tables
            def create_metrics_table(metrics_dict):
                return html.Table(
                    [html.Tr([html.Th('Model')] + [html.Th(metric) for metric in ['RMSE', 'MAE', 'MAPE', 'R²']])] +
                    [html.Tr([html.Td(model)] + 
                            [html.Td(f'{metrics[metric]:.3f}') for metric in ['RMSE', 'MAE', 'MAPE', 'R²']])
                     for model, metrics in metrics_dict.items()],
                    style={'width': '100%', 'border-collapse': 'collapse', 'text-align': 'center'}
                )
            
            emissions_table = create_metrics_table(emissions_metrics)
            temperature_table = create_metrics_table(temp_metrics)
            
            return fig, emissions_table, temperature_table

 
    def run_server(self, debug=True):
        self.app.run_server(debug=debug)

if __name__ == "__main__":
    data = pd.read_csv('./assets/processed_data.csv')
    dashboard = ClimateDashboard(data)
    dashboard.run_server(debug=False)