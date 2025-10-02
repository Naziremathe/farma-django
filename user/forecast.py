import pandas as pd
from prophet import Prophet

def prepare_data(df):
    # Prophet expects 'ds' and 'y' columns
    # Forecast price
    price_df = df[['date', 'average_price']].rename(columns={'date': 'ds', 'average_price': 'y'})
    
    # Forecast demand
    demand_df = df[['date', 'total_quantity_sold']].rename(columns={'date': 'ds', 'total_quantity_sold': 'y'})
    
    return price_df, demand_df

def forecast_prophet(df, periods=7):
    # df: daily summary dataframe with 'date', 'average_price', 'total_quantity_sold'
    price_df, demand_df = prepare_data(df)
    
    # Price forecast
    m_price = Prophet()
    m_price.fit(price_df)
    future_price = m_price.make_future_dataframe(periods=periods)
    forecast_price = m_price.predict(future_price)
    
    # Demand forecast
    m_demand = Prophet()
    m_demand.fit(demand_df)
    future_demand = m_demand.make_future_dataframe(periods=periods)
    forecast_demand = m_demand.predict(future_demand)
    
    # Return only the next 'periods' rows
    forecast_price = forecast_price[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    forecast_demand = forecast_demand[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    
    return forecast_price, forecast_demand


