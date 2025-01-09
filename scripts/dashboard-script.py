import dash
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ClimateDashboard:
    def __init__(self, data_path, forecasts_path, emissions_metrics_path, temp_metrics_path):
        self.app = dash.Dash(__name__)
        self.data = pd.read_csv(data_path)
        self.forecasts = pd.read_csv(forecasts_path)
        self.emissions_metrics = pd.read_csv(emissions_metrics_path, index_col=0)
        self.temp_metrics = pd.read_csv(temp_metrics_path, index_col=0)
        
        self.metrics = ['Emissions', 'GDP_per_capita', 'Renewable_Share', 'Energy_per_capita', 'Population', 'Emissions_per_GDP', 'Emissions_per_capita']
        
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
        table_style = {'width': '45%', 'display': 'inline-block', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'padding': '10px', 'margin': '10px'}
        div_style = {'backgroundColor': '#ffffff', 'padding': '20px', 'margin': '20px', 'borderRadius': '10px'}

        self.app.layout = html.Div([
            html.H1("Exploring Climate Trends and Forecasting Models"),

            # Intro Section
            html.Div([
                html.Div([
                    html.P(
                        """
                        I started this project to better understand the impact of climate change through data while also deepening my knowledge of time series forecasting models like ARIMA and Prophet. My goal was to explore how these models work and present these insights in an intuitive and interactive dashboard.
                        """
                    ),
                    html.P(
                        """
                        The dataset spans from 1970 to 2023, covering ten countries and global statistics. It includes metrics such as greenhouse gas (GHG) emissions, GDP per capita, renewable energy consumption, global temperatures, and more. 
                        Through this project, I also aimed to strengthen my skills in building dashboards using Python and Dash while providing clear visualizations and meaningful comparisons.
                        """
                    ),
                    html.P(
                        "This dashboard includes several sections to break down the data and models effectively:"
                    ),
                    html.Ul([
                        html.Li( [html.B("Time Series Visualizations: "), "Plotting all metrics for the ten selected countries to analyze trends over time."]),
                        html.Li( [html.B("Comparative Analysis: "), " Interactive visuals to compare climate-related metrics across the ten countries."]),
                        html.Li( [html.B("Forecasting Models Explained:"), "A breakdown of ARIMA and Prophet"]),
                        html.Li( [html.B("Forecast Model Comparison: "), "Time series analysis and forecasts for global emissions and temperature trends, highlighting the results of both models."]),
                        html.Li( [html.B("Model Comparison and Conclusions: "), "A performance evaluation of ARIMA and Prophet models with a discussion on their strengths and limitations."]),
                        
                    ]),
                    
                    html.P(
                        "Data sources: Our World in Data, International Monetary Fund, The World Bank, and The Emissions Database for Global Atmospheric Research report.",
                        style={'fontStyle': 'italic', 'fontSize': 'smaller'}
                    )
                ], style={'textAlign': 'left', 'padding': '20px', 'margin': '20px'})
            ], style={ 'padding': '20px', 'margin': '20px', 'textAlign': 'left'}),
            
            # Time Series Section
            html.Div([
                html.H2("Time Series Visualizations: Tracking Key Metrics"),
                html.Div([
                    html.Div([
                        html.Label("Select Country"),
                        dcc.Dropdown(
                            id='country-selector',
                            options=[{'label': c, 'value': c} 
                                   for c in sorted(self.data['Country'].unique())],
                            value='United States',
                            style=dropdown_style, 
                            searchable= False
                        )
                    ], style={'width': '30%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Select Metric"),
                        dcc.Dropdown(
                            id='metric-selector',
                            options=[{'label': m.replace('_', ' '), 'value': m} 
                                   for m in self.metrics],
                            value='Emissions',
                            style=dropdown_style, 
                            searchable= False
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'margin': '20px'}),
                ]),
                
                dcc.Graph(id='time-series-plot'),
            ], style=div_style),
            

            # Comparative Analysis of Metrics by Country
            html.Div([
                html.H2("Comparative Analysis of Metrics Across Countries"),
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
            ], style=div_style),


            # Explanation Section
            html.Div([
                html.H3("What Are Prophet and ARIMA Models?"),
                html.Div([
                    html.H4("Prophet Model:"),
                    html.P(
                        """
                            Prophet, developed by Facebook, is a user-friendly tool for forecasting time series with strong trends and seasonality. It decomposes data into three components: long-term trends, recurring patterns (like yearly cycles), and event effects (e.g., holidays). This flexibility makes it ideal for handling irregularities like missing data and outliers, with applications in sales, traffic, climate forecasting, and more. Minimal tuning and robust performance make Prophet popular across diverse fields.
                        """
                    ),
                    html.H4("ARIMA Model:"),
                    html.P(
                        """
                            The AutoRegressive Integrated Moving Average (ARIMA) model is a statistical approach to forecasting stationary time series. It relies on three components: auto-regression (using past values), differencing (to make data stationary), and moving average (using past errors). ARIMA is effective for stable, linear datasets and can handle seasonality with its extended form, SARIMA. However, it requires careful parameter selection, making it less user-friendly than Prophet.
                        """
                    ),
                    html.P(html.B("ARIMA’s parameters are defined as (p, d, q):")),
                    html.Ul([
                        html.Li( [html.B("p (Auto-Regressive Order): "), "The number of past observations used for prediction."]),
                        html.Li( [html.B("d (Differencing Order): "), "The number of times the data is differenced to make it stationary."]),
                        html.Li( [html.B("q (Moving Average Order): "), "The number of past forecast errors included in the model."]),
                    ]),
                    
                ], style={'textAlign': 'left', 'padding': '20px', 'margin': '20px'})
            ], style=div_style),   
            


            # World Forecast Section
            html.Div([
                html.H2("Forecast Model Comparison: Emissions and Temperature"),
                 html.Div([
                    html.P(
                        """
                           I utilized two forecasting models—Prophet and ARIMA—to predict global emissions and temperature trends. The ARIMA models were configured using auto.arima in R, which automatically selects the optimal model orders based on the data characteristics. Here's a breakdown of the models used:
                        """
                    ),
                    html.Ul([
                        html.Li( [html.B("Emissions Forecast (ARIMA (0, 2, 1)): "), "The model does not use past values (p = 0), applies second-order differencing (d = 2) to remove trends, and incorporates one lagged forecast error (q = 1)."]),
                        html.Li( [html.B("Temperature Forecast (ARIMA (3, 1, 0)): "), "This model uses the last three temperature values (p = 3), applies first-order differencing (d = 1) to make the data stationary, and does not include a moving average component (q = 0)."]),
                    ]),
                ], style={'textAlign': 'left', 'padding': '20px', 'margin': '20px'}),
                dcc.Graph(id='world-forecast')
            ], style=div_style),

            
            # Model Performance Comparison
            html.Div([
                html.H2("Model Comparison and Insights"),

                # Model Metrics Tables
                html.Div([
                    html.Div([
                        html.H4("Emissions Forecast Metrics"),
                        html.Div(id='emissions-metrics-table')
                    ], style=table_style),
                    
                    html.Div([
                        html.H4("Temperature Forecast Metrics"),
                        html.Div(id='temperature-metrics-table')
                    ], style=table_style)
                ]),
                html.Div([
                    html.H4("Understanding the Metrics:"),
                    html.Ul([
                        html.Li( [html.B("RMSE (Root Mean Square Error): "), "Measures the average prediction error, with larger errors penalized more heavily. Lower values indicate better accuracy."]),
                        html.Li( [html.B("MAE (Mean Absolute Error): "), "The average of absolute differences between predicted and actual values, less sensitive to large deviations compared to RMSE."]),
                        html.Li( [html.B("MAPE (Mean Absolute Percentage Error): "), "Indicates the percentage error relative to the observed values. Lower percentages indicate better performance."]),
                        html.Li( [html.B("R² (Coefficient of Determination): "), "Explains how much of the variance in the data is captured by the model. Values closer to 1 indicate a better fit."]),
                    ]),
                    html.H4("Model Insights"),
                    html.Ul([
                        html.Li( [html.B("Emissions Forecast: "), "Prophet performs significantly better, with lower error metrics (RMSE, MAE, and MAPE) and a much higher R² (0.995 vs. 0.814). This shows Prophet’s strength in modeling complex trends and variability."]),
                        html.Li( [html.B("Temperature Forecast: "), "While both models perform similarly, Prophet has a slight edge in accuracy. However, the high MAPE (50%+) for both models suggests a need for further refinement in forecasting temperature trends."]),
                    ]),
                    html.H4("Conclusion"),
                    html.P("Prophet generally outperforms ARIMA for both emissions and temperature forecasting. However, the high MAPE for temperature suggests that the models could benefit from refinement."),
                    html.P("This comparison is limited to Prophet and ARIMA, and other models, such as LSTMs or hybrid approaches, may be better suited for these tasks. Additionally, ARIMA can be further optimized by experimenting with different orders or preprocessing techniques."),
                ], style={'textAlign': 'left', 'padding': '20px', 'margin': '20px'})
            ], style=div_style)
        ], style={'maxWidth': '1200px', 'margin': 'auto', 'padding': '20px'})

    def setup_callbacks(self):
        @self.app.callback(
            Output('time-series-plot', 'figure'),
            [Input('country-selector', 'value'), 
             Input('metric-selector', 'value')]
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
            [Output('world-forecast', 'figure'),
             Output('emissions-metrics-table', 'children'),
             Output('temperature-metrics-table', 'children')],
            [Input('metric-selector', 'value')]
        )
        def update_forecast_and_metrics(_):
            # Create subplots
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('World Emissions Forecast',
                                            'Global Temperature Forecast'))
            
            # Convert dates
            self.forecasts['Date'] = pd.to_datetime(self.forecasts['Date'])
            last_historical_date = self.data[self.data['Country'] == 'World']['Year'].max()
            
            # Plot emissions
            world_data = self.data[self.data['Country'] == 'World']
            historical_dates = pd.to_datetime(world_data['Year'], format='%Y')
            
            fig.add_trace(
                go.Scatter(x=historical_dates, 
                          y=world_data['Emissions'],
                          name='Historical Emissions',
                          line=dict(color='blue')), row=1, col=1)
            
            fig.add_trace(
                go.Scatter(x=self.forecasts['Date'],
                          y=self.forecasts['Prophet_Emissions'],
                          name='Prophet Forecast',
                          line=dict(color='red', dash='dash')), row=1, col=1)
            
            fig.add_trace(
                go.Scatter(x=self.forecasts['Date'],
                          y=self.forecasts['ARIMA_Emissions'],
                          name='ARIMA Forecast',
                          line=dict(color='green', dash='dash')), row=1, col=1)
            
            # Plot temperatures
            fig.add_trace(
                go.Scatter(x=historical_dates,
                          y=world_data['Global_Temperature'],
                          name='Historical Temperature',
                          line=dict(color='orange')), row=2, col=1)
            
            fig.add_trace(
                go.Scatter(x=self.forecasts['Date'],
                          y=self.forecasts['Prophet_Temperature'],
                          name='Prophet Forecast',
                          line=dict(color='red', dash='dash')), row=2, col=1)
            
            fig.add_trace(
                go.Scatter(x=self.forecasts['Date'],
                          y=self.forecasts['ARIMA_Temperature'],
                          name='ARIMA Forecast',
                          line=dict(color='green', dash='dash')), row=2, col=1)
            
            fig.update_layout(height=1000, showlegend=True)
            
            # Create metrics tables
            def create_metrics_table(metrics_df):
                return html.Table(
                    [html.Tr([html.Th('Model')] + 
                            [html.Th(col) for col in metrics_df.columns])] +
                    [html.Tr([html.Td(index)] + 
                            [html.Td(f'{value:.3f}') for value in row])
                     for index, row in metrics_df.iterrows()],
                    style={'width': '100%', 'border-collapse': 'collapse', 'text-align': 'center'}
                )
            
            emissions_table = create_metrics_table(self.emissions_metrics)
            temperature_table = create_metrics_table(self.temp_metrics)
            
            return fig, emissions_table, temperature_table

    def run_server(self, debug=True, port=8050):
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    dashboard = ClimateDashboard(
        data_path='./assets/processed_data.csv',
        forecasts_path='./output/forecasts.csv',
        emissions_metrics_path='./output/emissions_metrics.csv',
        temp_metrics_path='./output/temperature_metrics.csv'
    )
    dashboard.run_server(debug=True)