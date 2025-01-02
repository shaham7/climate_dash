import dash
import pandas as pd 
import numpy as np
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ClimateDashboard:
    """
    Interactive dashboard for climate change analysis
    Combines time series analysis and economic-environmental metrics
    """
    
    def __init__(self, data, forecasts):
        self.app = dash.Dash(__name__)
        self.data = data
        self.forecasts = forecasts
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """
        Create the dashboard layout with multiple interactive components
        """
        self.app.layout = html.Div([
            html.H1("Climate Change Analysis Dashboard",
                   style={'textAlign': 'center'}),
            
            # Filters
            html.Div([
                html.Div([
                    html.Label("Select Country"),
                    dcc.Dropdown(
                        id='country-selector',
                        options=[{'label': c, 'value': c} 
                                for c in self.data['Country'].unique()],
                        value='World'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Select Metric"),
                    dcc.Dropdown(
                        id='metric-selector',
                        options=[
                            {'label': 'Emissions', 'value': 'Emissions'},
                            {'label': 'GDP per Capita', 'value': 'GDP_per_capita'},
                            {'label': 'Temperature', 'value': 'Global_Temperature'}
                        ],
                        value='Emissions'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Date Range"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=self.data['Year'].min(),
                        max=self.data['Year'].max(),
                        value=[self.data['Year'].min(), self.data['Year'].max()],
                        marks={str(year): str(year) 
                               for year in range(self.data['Year'].min(),
                                               self.data['Year'].max()+1, 5)}
                    )
                ], style={'width': '30%', 'display': 'inline-block'})
            ], style={'padding': '20px'}),
            
            # Graphs
            html.Div([
                # Time Series Plot
                dcc.Graph(id='time-series-plot'),
                
                # Correlation Plot
                dcc.Graph(id='correlation-plot'),
                
                # Comparative Analysis
                dcc.Graph(id='comparative-plot')
            ]),
            
            # Forecast Section
            html.Div([
                html.H2("Forecasting Results",
                        style={'textAlign': 'center'}),
                dcc.Graph(id='forecast-plot')
            ])
        ])
    
    def setup_callbacks(self):
        """
        Set up interactive callbacks for the dashboard
        """
        @self.app.callback(
            [Output('time-series-plot', 'figure'),
             Output('correlation-plot', 'figure'),
             Output('comparative-plot', 'figure'),
             Output('forecast-plot', 'figure')],
            [Input('country-selector', 'value'),
             Input('metric-selector', 'value'),
             Input('year-slider', 'value')]
        )
        def update_graphs(country, metric, years):
            # Filter data
            filtered_data = self.data[
                (self.data['Country'] == country) &
                (self.data['Year'].between(years[0], years[1]))
            ]
            
            # Time Series Plot
            ts_fig = px.line(filtered_data, x='Year', y=metric,
                           title=f'{metric} Over Time - {country}')
            
            # Correlation Plot
            corr_data = filtered_data[[
                'Emissions', 'GDP_per_capita', 'Energy_per_capita', 
                'Renewable_Share'
            ]].corr()
            corr_fig = px.imshow(corr_data, 
                                title=f'Correlation Matrix - {country}')
            
            # Comparative Plot
            latest_year = filtered_data['Year'].max()
            comp_data = self.data[self.data['Year'] == latest_year]
            comp_fig = px.bar(comp_data, x='Country', y=metric,
                            title=f'{metric} Comparison ({latest_year})')
            
            # Forecast Plot
            forecast_data = self.forecasts.get(country, {}).get(metric, None)
            if forecast_data is not None:
                forecast_fig = go.Figure()
                forecast_fig.add_trace(go.Scatter(
                    x=filtered_data['Year'],
                    y=filtered_data[metric],
                    name='Historical'
                ))
                forecast_fig.add_trace(go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat'],
                    name='Forecast'
                ))
                forecast_fig.update_layout(
                    title=f'{metric} Forecast - {country}'
                )
            else:
                forecast_fig = go.Figure()
            
            return ts_fig, corr_fig, comp_fig, forecast_fig
    
    def run_server(self, debug=True):
        """
        Run the dashboard server
        """
        self.app.run_server(debug=debug, host='127.0.0.1', port=8050)


if __name__ == "__main__":
    
    # Load the data
    data = pd.read_csv('processed_data.csv')
    
    # Create dummy forecast data (replace with real forecast data)
    np.random.seed(42)  # for reproducibility
    forecasts = pd.DataFrame({
        'ds': pd.date_range('2024-01-01', periods=5, freq='YE'),
        'yhat': [100, 110, 120, 130, 140],
        'yhat_lower': [90, 100, 110, 120, 130],  # Example lower bounds
        'yhat_upper': [110, 120, 130, 140, 150]   # Example upper bounds
    })

    
    # Initialize the dashboard
    dashboard = ClimateDashboard(data, forecasts)
    
    # Run the server
    dashboard.run_server(debug=True)
