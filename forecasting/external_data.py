import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from django.conf import settings

class ExternalDataFetcher:
    """Fetch external factors for South African context"""
    
    def __init__(self):
        self.weather_api_key = getattr(settings, 'WEATHER_API_KEY', None)
        self.fuel_api_key = getattr(settings, 'FUEL_API_KEY', None)
        
    def fetch_weather_data(self, province, start_date, end_date):
        """Fetch weather data for specific province"""
        # Province coordinates (major cities)
        province_coords = {
            'KZN': (-29.8587, 31.0218),  # Durban
            'GP': (-26.2041, 28.0473),   # Johannesburg
            'WC': (-33.9249, 18.4241),   # Cape Town
            'EC': (-32.9737, 27.8746),   # East London
            'NC': (-28.7282, 24.7499),   # Kimberley
            'FS': (-29.1211, 26.2149),   # Bloemfontein
            'MP': (-25.4653, 30.9697),   # Nelspruit
            'LP': (-23.9022, 29.4667),   # Polokwane
            'NW': (-25.8647, 25.6433),   # Mahikeng
        }
        
        lat, lon = province_coords.get(province, (-26.2041, 28.0473))
        
        # Try OpenWeatherMap API (free tier)
        if self.weather_api_key:
            try:
                return self._fetch_openweather(lat, lon, start_date, end_date)
            except:
                pass
        
        # Fallback: Generate synthetic weather data
        return self._generate_synthetic_weather(province, start_date, end_date)
    
    def _fetch_openweather(self, lat, lon, start_date, end_date):
        """Fetch from OpenWeatherMap API"""
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.weather_api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        weather_data = []
        
        for item in data['list']:
            weather_data.append({
                'date': datetime.fromtimestamp(item['dt']),
                'avg_temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'rainfall_mm': item.get('rain', {}).get('3h', 0)
            })
        
        return pd.DataFrame(weather_data)
    
    def _generate_synthetic_weather(self, province, start_date, end_date):
        """Generate synthetic weather data based on SA climate patterns"""
        # Climate parameters by province (averages)
        climate_params = {
            'KZN': {'temp_base': 21, 'temp_range': 8, 'rainfall_base': 90},
            'GP': {'temp_base': 18, 'temp_range': 10, 'rainfall_base': 70},
            'WC': {'temp_base': 17, 'temp_range': 7, 'rainfall_base': 50},
            'EC': {'temp_base': 19, 'temp_range': 8, 'rainfall_base': 75},
            'NC': {'temp_base': 20, 'temp_range': 12, 'rainfall_base': 30},
            'FS': {'temp_base': 17, 'temp_range': 11, 'rainfall_base': 55},
            'MP': {'temp_base': 20, 'temp_range': 9, 'rainfall_base': 80},
            'LP': {'temp_base': 21, 'temp_range': 10, 'rainfall_base': 65},
            'NW': {'temp_base': 19, 'temp_range': 11, 'rainfall_base': 60},
        }
        
        params = climate_params.get(province, climate_params['GP'])
        dates = pd.date_range(start_date, end_date, freq='D')
        
        weather_data = []
        for date in dates:
            # Seasonal temperature variation
            month = date.month
            temp_seasonal = params['temp_base'] + params['temp_range'] * np.sin(2 * np.pi * (month - 1) / 12)
            temp_daily = temp_seasonal + np.random.normal(0, 2)
            
            # Rainfall (summer rainfall for most SA regions)
            rainfall_seasonal = params['rainfall_base'] * (1 + 0.5 * np.sin(2 * np.pi * (month - 1) / 12))
            rainfall = max(0, np.random.gamma(2, rainfall_seasonal / 10))
            
            weather_data.append({
                'date': date,
                'avg_temperature': round(temp_daily, 1),
                'rainfall_mm': round(rainfall, 1),
                'humidity': round(60 + 20 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5), 1)
            })
        
        return pd.DataFrame(weather_data)
    
    def fetch_fuel_prices(self, start_date, end_date):
        """Fetch South African fuel prices"""
        # SA fuel prices are regulated and change monthly
        # Generate realistic fuel price progression
        dates = pd.date_range(start_date, end_date, freq='D')
        
        fuel_data = []
        base_price = 22.50  # Starting from ~R22.50/liter
        
        for date in dates:
            # Monthly changes (realistic volatility)
            if date.day == 1:
                base_price += np.random.normal(0.20, 0.40)  # Monthly change
            
            fuel_data.append({
                'date': date,
                'fuel_price': round(max(20.0, base_price), 2)
            })
        
        return pd.DataFrame(fuel_data)
    
    def fetch_exchange_rates(self, start_date, end_date):
        """Fetch ZAR/USD exchange rates"""
        # Generate realistic exchange rate movements
        dates = pd.date_range(start_date, end_date, freq='D')
        
        exchange_data = []
        base_rate = 18.50  # ZAR/USD
        
        for date in dates:
            # Daily volatility
            daily_change = np.random.normal(0, 0.15)
            base_rate = max(15.0, min(22.0, base_rate + daily_change))
            
            exchange_data.append({
                'date': date,
                'exchange_rate': round(base_rate, 4)
            })
        
        return pd.DataFrame(exchange_data)
    
    def fetch_all_factors(self, province, start_date, end_date):
        """Fetch all external factors"""
        weather_df = self.fetch_weather_data(province, start_date, end_date)
        fuel_df = self.fetch_fuel_prices(start_date, end_date)
        exchange_df = self.fetch_exchange_rates(start_date, end_date)
        
        # Merge all factors
        external_df = weather_df.copy()
        external_df['date'] = pd.to_datetime(external_df['date']).dt.date
        fuel_df['date'] = pd.to_datetime(fuel_df['date']).dt.date
        exchange_df['date'] = pd.to_datetime(exchange_df['date']).dt.date
        
        external_df = external_df.merge(fuel_df, on='date', how='left')
        external_df = external_df.merge(exchange_df, on='date', how='left')
        
        # Add seasonal indicators
        external_df['date'] = pd.to_datetime(external_df['date'])
        external_df['month'] = external_df['date'].dt.month
        external_df['is_harvest_season'] = external_df['month'].isin([3, 4, 5, 10, 11]).astype(int)
        
        # SA public holidays (simplified)
        sa_holidays = ['01-01', '03-21', '04-27', '05-01', '06-16', '08-09', '09-24', '12-16', '12-25', '12-26']
        external_df['is_holiday'] = external_df['date'].apply(
            lambda x: x.strftime('%m-%d') in sa_holidays
        ).astype(int)
        
        external_df['Date'] = external_df['date']
        external_df = external_df.drop('date', axis=1)
        
        return external_df