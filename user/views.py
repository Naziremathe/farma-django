from rest_framework import status, generics
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from django.db import transaction
from django.core.cache import cache
from django.shortcuts import render, redirect
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
from django.utils.decorators import method_decorator
from django.views.decorators.cache import never_cache
from django.urls import reverse_lazy
from django.views.generic import CreateView, ListView, DetailView, UpdateView
from django.shortcuts import get_object_or_404
from .forms import CropListingForm
from .models import CropListing
#import pandas as pd
from rest_framework.decorators import api_view
from .forecast import forecast_prophet
from .openai_utils import gpt_insights
import openai
from prophet import Prophet
#import numpy as np
from datetime import datetime, timedelta
import json
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import PasswordChangeForm
from .models import CustomUser
from .serializers import UserRegistrationSerializer, UserLoginSerializer, UserProfileSerializer, ChangePasswordSerializer
from rest_framework.views import APIView
from django.contrib.auth import update_session_auth_hash


OPENAI_API_KEY= ''

class UserRegistrationView(generics.CreateAPIView):
    permission_classes = [AllowAny]
    serializer_class = UserRegistrationSerializer

    @transaction.atomic
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        
        response_data = {
            'message': 'User registered successfully',
            'user': {
                'id': user.id,
                'email': user.email,
                'business_name': user.business_name,
            },
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        }
        
        return Response(response_data, status=status.HTTP_201_CREATED)

class UserLoginView(generics.GenericAPIView):
    permission_classes = [AllowAny]
    serializer_class = UserLoginSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        
        user = serializer.validated_data['user']
        refresh = RefreshToken.for_user(user)
        
        response_data = {
            'message': 'Login successful',
            'user': {
                'id': user.id,
                'email': user.email,
                'business_name': user.business_name,
            },
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        }
        
        return Response(response_data, status=status.HTTP_200_OK)


class SignUpView(View):
    @method_decorator(never_cache)
    def get(self, request):
        if request.user.is_authenticated:
            return redirect('forecast-dashboard')
        return render(request, 'user/index.html')
    
    @method_decorator(csrf_protect)
    def post(self, request):
        try:
            
            email = request.POST.get('email')
            phone_number = request.POST.get('phone_number')
            business_name = request.POST.get('business_name')
            province = request.POST.get('province')
            password = request.POST.get('password')
            confirm_password = request.POST.get('confirm_password')
            terms = request.POST.get('terms')
            
            
            if password != confirm_password:
                messages.error(request, "Passwords don't match.")
                return render(request, 'user/index.html', {
                    'email': email,
                    'phone_number': phone_number,
                    'business_name': business_name,
                    'province': province
                })
            
            if not terms:
                messages.error(request, "You must agree to the terms and conditions.")
                return render(request, 'user/index.html', {
                    'email': email,
                    'phone_number': phone_number,
                    'business_name': business_name,
                    'province': province
                })
            
            
            user = CustomUser.objects.create_user(
                email=email,
                phone_number=phone_number,
                business_name=business_name,
                province=province,
                password=password
            )
            
            
            backend = 'user.backends.EmailBackend'  
            login(request, user, backend=backend)
            
            messages.success(request, f'Account created successfully! Welcome, {business_name}!')
            return redirect('forecast-dashboard')
            
        except Exception as e:
            messages.error(request, f'Error creating account: {str(e)}')
            return render(request, 'user/index.html', {
                'email': email,
                'phone_number': phone_number,
                'business_name': business_name,
                'province': province
            })
        
class LoginView(View):
    @method_decorator(never_cache)
    @method_decorator(csrf_protect)
    def get(self, request):
        if request.user.is_authenticated:
            return redirect('forecast-dashboard')
        return render(request, 'user/login.html')
    
    @method_decorator(csrf_protect)
    def post(self, request):
        email = request.POST.get('email')
        password = request.POST.get('password')
        remember_me = request.POST.get('remember_me')
        
        
        user = authenticate(request, username=email, password=password)
        
        if user is not None:
            login(request, user)
            
            
            if not remember_me:
                request.session.set_expiry(0)  
            
            next_url = request.GET.get('next', 'forecast-dashboard')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid email or password. Please try again.')
            return render(request, 'user/login.html', {'email': email})


class LogoutView(View):
    def get(self, request):
        logout(request)
        return redirect('login')
    


class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]
    
    def get_object(self):
        return self.request.user
    
    def get(self, request, *args, **kwargs):
        if request.accepted_renderer.format == 'html' or 'text/html' in request.META.get('HTTP_ACCEPT', ''):
            
            return render(request, 'user/profile.html')  
        else:
            
            return super().get(request, *args, **kwargs)

class ChangePasswordView(APIView):
    """
    POST: Change user password
    """
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        serializer = ChangePasswordSerializer(
            data=request.data,
            context={'request': request}
        )
        
        if serializer.is_valid():
            serializer.save()
            
            update_session_auth_hash(request, request.user)
            
            return Response({
                'status': 'success',
                'message': 'Password updated successfully!'
            }, status=status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]
    
    def get_object(self):
        return self.request.user
    
    def get(self, request, *args, **kwargs):
        if request.accepted_renderer.format == 'html' or 'text/html' in request.META.get('HTTP_ACCEPT', ''):
            
            return render(request, 'user/profile.html', {'user': request.user})
        else:
            
            return super().get(request, *args, **kwargs)

@api_view(['POST'])
@login_required
def change_password(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.data)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)  
            return Response({
                'success': True,
                'message': 'Password updated successfully'
            })
        else:
            return Response({
                'success': False,
                'error': form.errors
            }, status=status.HTTP_400_BAD_REQUEST)

@api_view(['DELETE'])
@login_required
def delete_account(request):
    try:
        user = request.user
        user.delete()
        return Response({
            'success': True,
            'message': 'Account deleted successfully'
        })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_400_BAD_REQUEST)


class ForgotPasswordView(View):
    @method_decorator(csrf_protect)
    def get(self, request):
        return render(request, 'user/forgot_password.html')

    @method_decorator(csrf_protect)
    def post(self, request):
        email = request.POST.get('email')
        try:
            user = CustomUser.objects.get(email=email)
            
            messages.success(request, "Password reset instructions have been sent to your email.")
            return redirect('login')
        except CustomUser.DoesNotExist:
            messages.error(request, "No account found with that email address.")
            return render(request, 'user/forgot_password.html', {'email': email})


CROP_PRICE_MAPPING = {
    'potatoes': {
        'data_file': 'kzn_potatoes_daily_totals_excl_sweet.csv',
        'category_file': 'kzn_potatoes_category_summary_excl_sweet.csv',
        'seasonal_factor': 1.0,
        'volatility': 0.15
    },
    'tomatoes': {
        'data_file': None,  
        'category_file': None,
        'base_price': 35.0,
        'seasonal_factor': 1.2,
        'volatility': 0.25
    },
    'carrots': {
        'data_file': None,
        'category_file': None,
        'base_price': 28.0,
        'seasonal_factor': 0.9,
        'volatility': 0.18
    },
    'onions': {
        'data_file': None,
        'category_file': None,
        'base_price': 22.0,
        'seasonal_factor': 1.1,
        'volatility': 0.20
    },
    'cabbage': {
        'data_file': None,
        'category_file': None,
        'base_price': 18.0,
        'seasonal_factor': 0.8,
        'volatility': 0.22
    },
    
    'apples': {
        'data_file': None,
        'category_file': None,
        'base_price': 45.0,
        'seasonal_factor': 1.3,
        'volatility': 0.20
    },
    'bananas': {
        'data_file': None,
        'category_file': None,
        'base_price': 25.0,
        'seasonal_factor': 1.0,
        'volatility': 0.15
    }
}

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def crop_price_prediction_api(request):
    """API endpoint to get price predictions for selected crops"""
    try:
        data = json.loads(request.body)
        crop_name = data.get('crop_name', '').lower()
        category = data.get('category', '').lower()
        
        if not crop_name or not category:
            return JsonResponse({
                'error': 'Missing crop_name or category'
            }, status=400)
        
        
        prediction = get_crop_price_prediction(crop_name, category)
        
        return JsonResponse(prediction)
        
    except Exception as e:
        print(f"Error in crop_price_prediction_api: {e}")
        return JsonResponse({
            'error': 'Unable to fetch price prediction',
            'message': str(e)
        }, status=500)

def get_crop_price_prediction(crop_name, category):
    """Get price prediction for a specific crop"""
    
    
    if crop_name in CROP_PRICE_MAPPING:
        crop_config = CROP_PRICE_MAPPING[crop_name]
        
        
        if crop_config.get('data_file') and crop_config.get('category_file'):
            return get_real_data_prediction(crop_name, crop_config)
        else:
            
            return get_synthetic_prediction(crop_name, crop_config, category)
    else:
        
        return get_default_prediction(crop_name, category)

def get_real_data_prediction(crop_name, crop_config):
    """Get prediction using real market data"""
    try:
        
        df1 = pd.read_csv(crop_config['category_file'])
        df2 = pd.read_csv(crop_config['data_file'])
        
        
        df = create_time_series_data_for_prediction(df1, df2)
        
        
        advanced_forecaster = AdvancedSAForecasting()
        
        
        forecasts = advanced_forecaster.forecast_with_external_factors(df, periods_7=7, periods_30=7)
        
        
        if '7_day' in forecasts and 'price' in forecasts['7_day']:
            price_forecast = forecasts['7_day']['price']
            
            if len(price_forecast) > 0:
                prices = price_forecast['yhat'].abs()  
                
                min_price = round(prices.min(), 2)
                max_price = round(prices.max(), 2)
                avg_price = round(prices.mean(), 2)
                
                
                if len(prices) > 1:
                    trend_slope = (prices.iloc[-1] - prices.iloc[0]) / len(prices)
                    if trend_slope > 0.5:
                        trend = "Rising"
                        trend_icon = "fa-arrow-up"
                        trend_color = "text-green-600"
                    elif trend_slope < -0.5:
                        trend = "Declining"
                        trend_icon = "fa-arrow-down"
                        trend_color = "text-red-600"
                    else:
                        trend = "Stable"
                        trend_icon = "fa-minus"
                        trend_color = "text-gray-600"
                else:
                    trend = "Stable"
                    trend_icon = "fa-minus"
                    trend_color = "text-gray-600"
                
                return {
                    'success': True,
                    'crop_name': crop_name.title(),
                    'min_price': min_price,
                    'max_price': max_price,
                    'avg_price': avg_price,
                    'trend': trend,
                    'trend_icon': trend_icon,
                    'trend_color': trend_color,
                    'data_source': 'Real market data',
                    'confidence': 'High'
                }
    
    except Exception as e:
        print(f"Error getting real data prediction for {crop_name}: {e}")
        
        return get_synthetic_prediction(crop_name, crop_config, 'vegetables')
    
    
    return get_default_prediction(crop_name, 'vegetables')


def get_synthetic_prediction(crop_name, crop_config, category):
    """Generate synthetic prediction based on market patterns"""
    
    base_price = crop_config.get('base_price', 30.0)
    seasonal_factor = crop_config.get('seasonal_factor', 1.0)
    volatility = crop_config.get('volatility', 0.20)
    
    
    current_month = datetime.now().month
    
    
    if category == 'vegetables':
        if current_month in [2, 3, 4]:  
            seasonal_multiplier = 0.85
        elif current_month in [9, 10, 11]:  
            seasonal_multiplier = 1.15
        else:
            seasonal_multiplier = 1.0
    elif category == 'fruits':
        if current_month in [1, 2, 12]:  
            seasonal_multiplier = 0.9
        elif current_month in [6, 7, 8]:  
            seasonal_multiplier = 1.2
        else:
            seasonal_multiplier = 1.0
    else:
        seasonal_multiplier = 1.0
    
    
    adjusted_base = base_price * seasonal_factor * seasonal_multiplier
    
    
    np.random.seed(hash(crop_name) % 1000)  
    price_variation = np.random.normal(0, adjusted_base * volatility)
    market_price = max(5.0, adjusted_base + price_variation)
    
    
    price_range = market_price * volatility
    min_price = round(max(5.0, market_price - price_range), 2)
    max_price = round(market_price + price_range, 2)
    avg_price = round(market_price, 2)
    
    
    if seasonal_multiplier > 1.05:
        trend = "Rising"
        trend_icon = "fa-arrow-up"
        trend_color = "text-green-600"
    elif seasonal_multiplier < 0.95:
        trend = "Declining"
        trend_icon = "fa-arrow-down"
        trend_color = "text-red-600"
    else:
        trend = "Stable"
        trend_icon = "fa-minus"
        trend_color = "text-gray-600"
    
    return {
        'success': True,
        'crop_name': crop_name.title(),
        'min_price': min_price,
        'max_price': max_price,
        'avg_price': avg_price,
        'trend': trend,
        'trend_icon': trend_icon,
        'trend_color': trend_color,
        'data_source': f'Market analysis for {category}',
        'confidence': 'Medium'
    }

def get_default_prediction(crop_name, category):
    """Default prediction for unknown crops"""
    
    
    category_base_prices = {
        'vegetables': 25.0,
        'fruits': 40.0,
        'livestock': 180.0  
    }
    
    base_price = category_base_prices.get(category, 30.0)
    
    
    np.random.seed(hash(crop_name) % 1000)
    price_multiplier = 0.8 + (np.random.random() * 0.4)  
    market_price = base_price * price_multiplier
    
    
    price_range = market_price * 0.25
    min_price = round(max(5.0, market_price - price_range), 2)
    max_price = round(market_price + price_range, 2)
    avg_price = round(market_price, 2)
    
    return {
        'success': True,
        'crop_name': crop_name.title(),
        'min_price': min_price,
        'max_price': max_price,
        'avg_price': avg_price,
        'trend': 'Stable',
        'trend_icon': 'fa-minus',
        'trend_color': 'text-gray-600',
        'data_source': 'Estimated based on category averages',
        'confidence': 'Low'
    }

def create_time_series_data_for_prediction(df1, df2):
    """Helper function to create time series data for price prediction"""
    
    
    price_stats = {
        'avg_price': df1['AvgPrice'].mean(),
        'price_std': df1['AvgPrice'].std(),
        'min_price': df1['AvgPrice'].min(),
        'max_price': df1['AvgPrice'].max()
    }
    
    
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2 = df2.sort_values('Date').reset_index(drop=True)
    
    
    result_data = []
    
    for idx, row in df2.iterrows():
        date = row['Date']
        quantity = row['KgSold'] if pd.notna(row['KgSold']) else 0
        
        
        day_of_week = date.weekday()
        weekend_multiplier = 1.05 if day_of_week >= 5 else 1.0
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
        
        np.random.seed(hash(str(date)) % 1000)
        price_variation = np.random.normal(0, price_stats['price_std'] * 0.3)
        
        estimated_price = (price_stats['avg_price'] + price_variation) * weekend_multiplier * seasonal_factor
        estimated_price = max(estimated_price, price_stats['min_price'] * 0.8)
        estimated_price = min(estimated_price, price_stats['max_price'] * 1.2)
        
        result_data.append({
            'date': date,
            'average_price': estimated_price,
            'total_quantity_sold': quantity
        })
    
    
    if len(result_data) < 10:
        start_date = df2['Date'].min() - timedelta(days=30)
        
        for i in range(30):
            date = start_date + timedelta(days=i)
            avg_quantity = df2['KgSold'].mean()
            quantity_variation = np.random.normal(0, df2['KgSold'].std() * 0.5)
            estimated_quantity = max(0, avg_quantity + quantity_variation)
            
            np.random.seed(hash(str(date)) % 1000)
            price_variation = np.random.normal(0, price_stats['price_std'] * 0.3)
            estimated_price = price_stats['avg_price'] + price_variation
            
            result_data.append({
                'date': date,
                'average_price': estimated_price,
                'total_quantity_sold': estimated_quantity
            })
    
    df_result = pd.DataFrame(result_data)
    df_result = df_result.sort_values('date').reset_index(drop=True)
    
    return df_result


class EnhancedCropUploadView(CreateView):
    model = CropListing
    form_class = CropListingForm
    template_name = 'user/crop_upload.html'

    def form_valid(self, form):
        form.instance.farmer = self.request.user
        form.instance.status = 'published'
        form.instance.unit = 'kg'
        messages.success(self.request, 'Crop information uploaded successfully!')
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy('crop_list')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'Upload Crop Information'
        return context

class CropListView(LoginRequiredMixin, ListView):
    model = CropListing
    template_name = 'user/crop_list.html'
    context_object_name = 'crops'
    paginate_by = 10

    def get_queryset(self):
        return CropListing.objects.filter(farmer=self.request.user).order_by('-created_at')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['page_title'] = 'My Crop Listings'
        return context







import pandas as pd
import numpy as np
from prophet import Prophet
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SouthAfricanMarketData:
    """Collect external factors relevant to SA agricultural markets"""
    
    def __init__(self):
        self.weather_api_key = None  
        self.cache = {}
        self.sa_holidays = self.get_sa_holidays()
    
    def get_sa_holidays(self):
        """South African public holidays that affect agricultural markets"""
        current_year = datetime.now().year
        holidays = [
            f"{current_year}-01-01",  
            f"{current_year}-03-21",  
            f"{current_year}-04-27",  
            f"{current_year}-05-01",  
            f"{current_year}-06-16",  
            f"{current_year}-08-09",  
            f"{current_year}-09-24",  
            f"{current_year}-12-16",  
            f"{current_year}-12-25",  
            f"{current_year}-12-26",  
        ]
        return pd.to_datetime(holidays)
    
    def get_weather_data(self, city="Durban"):
        """Get weather forecast for KZN (using Durban as reference)"""
        try:
            if not self.weather_api_key:
                
                return self.get_synthetic_weather_data()
            
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city},ZA&appid={self.weather_api_key}&units=metric"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                weather_data = []
                
                for item in data['list'][:40]:  
                    weather_data.append({
                        'date': pd.to_datetime(item['dt'], unit='s'),
                        'temperature': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'rainfall': item.get('rain', {}).get('3h', 0),
                        'weather_desc': item['weather'][0]['description']
                    })
                
                return pd.DataFrame(weather_data)
            
        except Exception as e:
            print(f"Weather API error: {e}")
            return self.get_synthetic_weather_data()
    
    def get_synthetic_weather_data(self):
        """Generate realistic weather data for KZN based on seasonal patterns"""
        dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
        weather_data = []
        
        for date in dates:
            
            month = date.month
            
            
            if month in [11, 12, 1, 2, 3]:
                base_temp = 26 + np.random.normal(0, 3)
                base_humidity = 75 + np.random.normal(0, 10)
                rainfall_prob = 0.4
            
            elif month in [6, 7, 8]:
                base_temp = 18 + np.random.normal(0, 2)
                base_humidity = 60 + np.random.normal(0, 8)
                rainfall_prob = 0.1
            
            else:
                base_temp = 22 + np.random.normal(0, 2)
                base_humidity = 65 + np.random.normal(0, 8)
                rainfall_prob = 0.25
            
            rainfall = np.random.exponential(5) if np.random.random() < rainfall_prob else 0
            
            weather_data.append({
                'date': date,
                'temperature': max(10, min(35, base_temp)),
                'humidity': max(30, min(95, base_humidity)),
                'rainfall': rainfall
            })
        
        return pd.DataFrame(weather_data)
    
    def get_fuel_prices(self):
        """Get SA fuel prices - using synthetic data (you can replace with actual API)"""
        
        
        
        base_price = 23.50  
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=60, freq='D')
        
        fuel_data = []
        for i, date in enumerate(dates):
            
            weekly_cycle = 0.2 * np.sin(2 * np.pi * i / 7)
            monthly_trend = 0.1 * np.sin(2 * np.pi * i / 30)
            random_shock = np.random.normal(0, 0.15)
            
            price = base_price + weekly_cycle + monthly_trend + random_shock
            fuel_data.append({
                'date': date,
                'fuel_price': max(20, min(30, price))  
            })
        
        return pd.DataFrame(fuel_data)
    
    def get_exchange_rate(self):
        """Get USD/ZAR exchange rate"""
        try:
            
            url = "https://api.exchangerate-api.com/v4/latest/USD"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current_rate = data['rates'].get('ZAR', 18.5)
                
                
                dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=60, freq='D')
                exchange_data = []
                
                for i, date in enumerate(dates):
                    
                    volatility = np.random.normal(0, 0.3)
                    rate = current_rate + volatility
                    exchange_data.append({
                        'date': date,
                        'usd_zar': max(15, min(22, rate))  
                    })
                
                return pd.DataFrame(exchange_data)
                
        except Exception as e:
            print(f"Exchange rate API error: {e}")
        
        
        return self.get_synthetic_exchange_data()
    
    def get_synthetic_exchange_data(self):
        """Generate realistic USD/ZAR exchange rate data"""
        base_rate = 18.5
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=60, freq='D')
        
        exchange_data = []
        for i, date in enumerate(dates):
            daily_volatility = np.random.normal(0, 0.25)
            weekly_pattern = 0.1 * np.sin(2 * np.pi * i / 7)
            
            rate = base_rate + daily_volatility + weekly_pattern
            exchange_data.append({
                'date': date,
                'usd_zar': max(16, min(21, rate))
            })
        
        return pd.DataFrame(exchange_data)
    
    def get_cpi_inflation_factor(self):
        """Get food inflation factor based on SA CPI trends"""
        
        base_inflation = 5.5  
        monthly_variation = np.random.normal(0, 0.5)
        
        current_factor = (100 + base_inflation + monthly_variation) / 100
        return max(1.02, min(1.12, current_factor))  
    
    def get_seasonal_factors(self, dates):
        """Calculate seasonal multipliers for potato prices in SA"""
        seasonal_data = []
        
        for date in dates:
            month = date.month
            day_of_year = date.dayofyear
            
            
            
            
            
            
            if month in [2, 3, 4]:  
                seasonal_multiplier = 0.85 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)
            elif month in [5, 6, 7, 8]:  
                seasonal_multiplier = 0.95 + 0.05 * np.sin(2 * np.pi * day_of_year / 365)
            else:  
                seasonal_multiplier = 1.1 + 0.15 * np.sin(2 * np.pi * day_of_year / 365)
            
            
            is_holiday = any(abs((date - holiday).days) <= 2 for holiday in self.sa_holidays)
            if is_holiday:
                seasonal_multiplier *= 1.05  
            
            seasonal_data.append({
                'date': date,
                'seasonal_factor': seasonal_multiplier,
                'is_holiday': is_holiday,
                'month': month
            })
        
        return pd.DataFrame(seasonal_data)


class AdvancedSAForecasting:
    """Advanced forecasting with South African market factors"""
    
    def __init__(self):
        self.market_data = SouthAfricanMarketData()
        self.external_factors_cache = {}
    
    def collect_external_factors(self, forecast_periods=30):
        """Collect all external factors for forecasting"""
        print("Collecting South African market factors...")
        
        try:
            
            weather_df = self.market_data.get_weather_data()
            
            
            fuel_df = self.market_data.get_fuel_prices()
            
            
            exchange_df = self.market_data.get_exchange_rate()
            
            
            forecast_dates = pd.date_range(
                start=datetime.now(), 
                periods=forecast_periods, 
                freq='D'
            )
            seasonal_df = self.market_data.get_seasonal_factors(forecast_dates)
            
            
            inflation_factor = self.market_data.get_cpi_inflation_factor()
            
            print(f"External factors collected:")
            print(f"- Weather data: {len(weather_df)} records")
            print(f"- Fuel prices: {len(fuel_df)} records") 
            print(f"- Exchange rates: {len(exchange_df)} records")
            print(f"- Seasonal factors: {len(seasonal_df)} records")
            print(f"- Current inflation factor: {inflation_factor:.3f}")
            
            return {
                'weather': weather_df,
                'fuel': fuel_df,
                'exchange': exchange_df,
                'seasonal': seasonal_df,
                'inflation': inflation_factor
            }
            
        except Exception as e:
            print(f"Error collecting external factors: {e}")
            return self.get_fallback_factors(forecast_periods)
    
    def get_fallback_factors(self, periods):
        """Provide fallback factors if external APIs fail"""
        dates = pd.date_range(start=datetime.now(), periods=periods, freq='D')
        
        return {
            'weather': pd.DataFrame({
                'date': dates,
                'temperature': 22,
                'humidity': 65,
                'rainfall': 0
            }),
            'fuel': pd.DataFrame({
                'date': dates,
                'fuel_price': 23.5
            }),
            'exchange': pd.DataFrame({
                'date': dates,
                'usd_zar': 18.5
            }),
            'seasonal': pd.DataFrame({
                'date': dates,
                'seasonal_factor': 1.0,
                'is_holiday': False,
                'month': dates[0].month
            }),
            'inflation': 1.05
        }
    
    def prepare_enhanced_data(self, df, external_factors):
        """
        Merge main dataset with external factors and prepare for forecasting.
        Fills any missing values and ensures all numeric columns are ready for Prophet.
        
        Args:
            df (pd.DataFrame): Your main dataframe with at least ['date', 'average_price', 'total_quantity_sold']
            external_factors (dict): Dictionary containing dataframes for weather, fuel, exchange rate, seasonal factors
        
        Returns:
            pd.DataFrame: Clean dataframe ready for Prophet
        """
        enhanced_df = df.copy()
        
        
        enhanced_df['date'] = pd.to_datetime(enhanced_df['date'])
        
        
        if 'weather' in external_factors:
            weather_df = external_factors['weather'].copy()
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            
            weather_df['weather_score'] = (
                (weather_df['temperature'] - weather_df['temperature'].mean()) / 10
                - (weather_df['rainfall'] / 10)
            )
            enhanced_df = enhanced_df.merge(weather_df[['date', 'weather_score']], on='date', how='left')
        
        
        if 'fuel' in external_factors:
            fuel_df = external_factors['fuel'].copy()
            fuel_df['date'] = pd.to_datetime(fuel_df['date'])
            enhanced_df = enhanced_df.merge(fuel_df[['date', 'fuel_price']], on='date', how='left')
        
        if 'exchange' in external_factors:
            exchange_df = external_factors['exchange'].copy()
            exchange_df['date'] = pd.to_datetime(exchange_df['date'])
            enhanced_df = enhanced_df.merge(exchange_df[['date', 'usd_zar']], on='date', how='left')
        
        
        if 'seasonal' in external_factors:
            seasonal_df = external_factors['seasonal'].copy()
            seasonal_df['date'] = pd.to_datetime(seasonal_df['date'])
            enhanced_df = enhanced_df.merge(
                seasonal_df[['date', 'seasonal_factor', 'is_holiday']], on='date', how='left'
            )
        
        
        enhanced_df['economic_score'] = (
            enhanced_df.get('fuel_price', 0).fillna(0) * 0.5
            + enhanced_df.get('usd_zar', 0).fillna(0) * 0.5
        )
        
        enhanced_df['market_score'] = enhanced_df.get('seasonal_factor', 0).fillna(0)
        
        
        enhanced_df['weather_score'] = enhanced_df.get('weather_score', 0).fillna(0)
        enhanced_df['economic_score'] = enhanced_df.get('economic_score', 0).fillna(0)
        enhanced_df['market_score'] = enhanced_df.get('market_score', 0).fillna(0)
        enhanced_df['is_holiday'] = enhanced_df.get('is_holiday', 0).fillna(0).astype(int)
        
        return enhanced_df

    
    def forecast_with_external_factors(self, df, periods_7=7, periods_30=30):
        """Create enhanced forecasts for 7 and 30 days with external factors"""
        
        try:
            
            external_factors = self.collect_external_factors(max(periods_7, periods_30))
            
            
            enhanced_df = self.prepare_enhanced_data(df, external_factors)

            
            for col in ['weather_score', 'economic_score', 'market_score', 'is_holiday']:
                if col not in enhanced_df.columns:
                    enhanced_df[col] = 0
                else:
                    enhanced_df[col] = enhanced_df[col].fillna(0)

            
            enhanced_df['is_holiday'] = enhanced_df['is_holiday'].astype(int)
            
            
            forecasts = {}
            
            for period_name, periods in [('7_day', periods_7), ('30_day', periods_30)]:
                print(f"\nGenerating {period_name} forecast...")
                
                try:
                    
                    price_forecast = self.create_price_forecast_with_factors(
                        enhanced_df, periods, external_factors
                    )
                except Exception as e:
                    print(f"Price forecast failed for {period_name}: {e}")
                    price_forecast = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])

                try:
                    
                    demand_forecast = self.create_demand_forecast_with_factors(
                        enhanced_df, periods, external_factors
                    )
                except Exception as e:
                    print(f"Demand forecast failed for {period_name}: {e}")
                    demand_forecast = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])

                forecasts[period_name] = {
                    'price': price_forecast,
                    'demand': demand_forecast,
                    'external_factors': external_factors
                }
                
                print(f"{period_name} forecast completed")
            
            return forecasts
            
        except Exception as e:
            print(f"Error in enhanced forecasting: {e}")
            import traceback
            traceback.print_exc()
            
            return self.fallback_forecast(df, periods_7, periods_30)

    def create_price_forecast_with_factors(self, enhanced_df, periods, external_factors):
        """Create price forecast incorporating external factors"""

        
        prophet_df = enhanced_df[['date', 'average_price']].rename(
            columns={'date': 'ds', 'average_price': 'y'}
        )

        
        for col in ['weather_score', 'economic_score', 'market_score']:
            if col in enhanced_df.columns:
                prophet_df[col] = enhanced_df[col].fillna(0)  
            else:
                prophet_df[col] = 0  

        
        prophet_df['is_holiday'] = enhanced_df.get('is_holiday', False)

        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays_prior_scale=10,
            seasonality_prior_scale=5,
            interval_width=0.8
        )

        
        for col in ['weather_score', 'economic_score', 'market_score', 'is_holiday']:
            if col in prophet_df.columns:
                model.add_regressor(col)

        
        model.fit(prophet_df)

        
        future = model.make_future_dataframe(periods=periods)

        
        future_factors = self.project_future_factors(external_factors, periods)

        
        future = future.merge(future_factors, left_on='ds', right_on='date', how='left')
        future = future.fillna(method='ffill').fillna(0)

        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

    def create_demand_forecast_with_factors(self, enhanced_df, periods, external_factors):
        """Create demand forecast incorporating external factors"""

        prophet_df = enhanced_df[['date', 'total_quantity_sold']].rename(
            columns={'date': 'ds', 'total_quantity_sold': 'y'}
        )

        
        prophet_df['weather_score'] = enhanced_df.get('weather_score', 0).fillna(0) * 0.5
        prophet_df['economic_score'] = enhanced_df.get('economic_score', 0).fillna(0) * -1
        prophet_df['market_score'] = enhanced_df.get('market_score', 0).fillna(0)

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_prior_scale=5,
            interval_width=0.8
        )

        for col in ['weather_score', 'economic_score', 'market_score']:
            model.add_regressor(col)

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=periods)
        future_factors = self.project_future_factors(external_factors, periods)
        future = future.merge(future_factors, left_on='ds', right_on='date', how='left')
        future = future.fillna(method='ffill').fillna(0)

        
        future['weather_score'] = future['weather_score'] * 0.5
        future['economic_score'] = future['economic_score'] * -1

        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

    
    def project_future_factors(self, external_factors, periods):
        """Project external factors into the future"""
        
        future_dates = pd.date_range(
            start=datetime.now(),
            periods=periods,
            freq='D'
        )
        
        future_data = []
        
        for date in future_dates:
            
            weather_row = external_factors['weather'].iloc[-1] if len(external_factors['weather']) > 0 else {}
            fuel_row = external_factors['fuel'].iloc[-1] if len(external_factors['fuel']) > 0 else {}
            exchange_row = external_factors['exchange'].iloc[-1] if len(external_factors['exchange']) > 0 else {}
            
            
            seasonal_row = external_factors['seasonal'][
                external_factors['seasonal']['date'].dt.date == date.date()
            ]
            
            if len(seasonal_row) > 0:
                seasonal_factor = seasonal_row.iloc[0]['seasonal_factor']
                is_holiday = seasonal_row.iloc[0]['is_holiday']
            else:
                seasonal_factor = 1.0
                is_holiday = False
            
            
            weather_score = (
                (weather_row.get('temperature', 22) - 20) / 10 +
                (weather_row.get('rainfall', 0) / 10) * -0.5
            )
            
            economic_score = (
                (fuel_row.get('fuel_price', 23.5) - 23) / 23 * -1 +
                (exchange_row.get('usd_zar', 18.5) - 18) / 18 * 0.5
            )
            
            future_data.append({
                'date': date,
                'weather_score': weather_score,
                'economic_score': economic_score,
                'market_score': seasonal_factor,
                'is_holiday': is_holiday
            })
        
        return pd.DataFrame(future_data)
    
    def fallback_forecast(self, df, periods_7, periods_30):
        """Simple fallback forecast without external factors"""
        
        print("Using fallback forecasting (no external factors)")
        
        
        price_df = df[['date', 'average_price']].rename(columns={'date': 'ds', 'average_price': 'y'})
        demand_df = df[['date', 'total_quantity_sold']].rename(columns={'date': 'ds', 'total_quantity_sold': 'y'})
        
        forecasts = {}
        
        for period_name, periods in [('7_day', periods_7), ('30_day', periods_30)]:
            
            price_model = Prophet()
            price_model.fit(price_df)
            price_future = price_model.make_future_dataframe(periods=periods)
            price_forecast = price_model.predict(price_future).tail(periods)
            
            
            demand_model = Prophet()
            demand_model.fit(demand_df)
            demand_future = demand_model.make_future_dataframe(periods=periods)
            demand_forecast = demand_model.predict(demand_future).tail(periods)
            
            forecasts[period_name] = {
                'price': price_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                'demand': demand_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                'external_factors': None
            }
        
        return forecasts



class EnhancedDashboardView(LoginRequiredMixin, View):
    login_url = '/login/'
    redirect_field_name = 'next'
    
    def create_time_series_data(self, df1, df2):
        """
        Create time series data from your CSV structure:
        - df1: Category summary (has prices but no dates)
        - df2: Daily totals (has dates but no prices)
        """
        
        print("Creating time series data...")
        print(f"DF1 (category data): {df1.shape}")
        print(f"DF2 (daily data): {df2.shape}")
        
        
        price_stats = {
            'avg_price': df1['AvgPrice'].mean(),
            'price_std': df1['AvgPrice'].std(),
            'min_price': df1['AvgPrice'].min(),
            'max_price': df1['AvgPrice'].max()
        }
        
        print(f"Price statistics from DF1: {price_stats}")
        
        
        df2['Date'] = pd.to_datetime(df2['Date'])
        df2 = df2.sort_values('Date').reset_index(drop=True)
        
        
        result_data = []
        
        for idx, row in df2.iterrows():
            
            date = row['Date']
            
            
            quantity = row['KgSold'] if pd.notna(row['KgSold']) else 0
            
            
            
            day_of_week = date.weekday()
            weekend_multiplier = 1.05 if day_of_week >= 5 else 1.0
            
            
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
            
            
            np.random.seed(hash(str(date)) % 1000)  
            price_variation = np.random.normal(0, price_stats['price_std'] * 0.3)
            
            estimated_price = (price_stats['avg_price'] + price_variation) * weekend_multiplier * seasonal_factor
            estimated_price = max(estimated_price, price_stats['min_price'] * 0.8)  
            estimated_price = min(estimated_price, price_stats['max_price'] * 1.2)  
            
            result_data.append({
                'date': date,
                'average_price': estimated_price,
                'total_quantity_sold': quantity
            })
        
        
        if len(result_data) < 10:  
            print("Extending data with historical estimates...")
            start_date = df2['Date'].min() - timedelta(days=30)  
            
            for i in range(30):
                date = start_date + timedelta(days=i)
                
                
                avg_quantity = df2['KgSold'].mean()
                quantity_variation = np.random.normal(0, df2['KgSold'].std() * 0.5)
                estimated_quantity = max(0, avg_quantity + quantity_variation)
                
                
                np.random.seed(hash(str(date)) % 1000)
                price_variation = np.random.normal(0, price_stats['price_std'] * 0.3)
                estimated_price = price_stats['avg_price'] + price_variation
                
                result_data.append({
                    'date': date,
                    'average_price': estimated_price,
                    'total_quantity_sold': estimated_quantity
                })
        
        
        df_result = pd.DataFrame(result_data)
        df_result = df_result.sort_values('date').reset_index(drop=True)
        
        print(f"Created time series with {len(df_result)} data points")
        print(f"Date range: {df_result['date'].min()} to {df_result['date'].max()}")
        
        return df_result
    
    def get(self, request):
        user = request.user

        try:
            df1 = pd.read_csv('kzn_potatoes_category_summary_excl_sweet.csv')
            df2 = pd.read_csv('kzn_potatoes_daily_totals_excl_sweet.csv')

            print("DF1 columns:", df1.columns.tolist())
            print("DF2 columns:", df2.columns.tolist())
            print("DF1 sample:")
            print(df1.head(3))
            print("DF2 sample:")
            print(df2.head(3))
            
            
            df = self.create_time_series_data(df1, df2)
            
            print("Final dataset sample:")
            print(df.head())
            print(f"Data shape: {df.shape}")

            
            advanced_forecaster = AdvancedSAForecasting()
            
            
            forecasts = advanced_forecaster.forecast_with_external_factors(df, periods_7=7, periods_30=30)
            
            print("Enhanced forecasts generated successfully")
            
            
            insights = self.generate_enhanced_insights(forecasts, df)
            
            
            forecast_price_formatted, forecast_demand_formatted = self.format_for_existing_template(forecasts['7_day'])
            
            
            forecast_summary = self.calculate_forecast_summary(forecasts)
            
            
            summary_stats = self.calculate_detailed_summary_statistics(forecasts, df)
            
            context = {
                'user': user,
                'business_name': user.business_name,
                'email': user.email,
                'phone_number': str(user.phone_number),
                'province': user.get_province_display(),
                'date_joined': user.date_joined.strftime('%B %d, %Y'),
                
                
                'forecast_price': forecast_price_formatted,
                'forecast_demand': forecast_demand_formatted,
                'insights': insights,
                
                
                'forecast_summary': forecast_summary,
                
                
                'forecast_7_day': self.format_forecasts_for_template(forecasts)['7_day'],
                'forecast_30_day': self.format_forecasts_for_template(forecasts)['30_day'],
                'summary_stats': summary_stats,
                
                
                'data_summary': {
                    'total_records': len(df),
                    'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
                    'avg_price': round(df['average_price'].mean(), 2),
                    'avg_quantity': round(df['total_quantity_sold'].mean(), 0),
                    'external_factors_used': forecasts.get('7_day', {}).get('external_factors') is not None
                }
            }
            
            return render(request, 'user/dashboard.html', context)
            
        except Exception as e:
            print(f"Enhanced dashboard error: {e}")
            import traceback
            traceback.print_exc()
            
            return render(request, 'user/dashboard.html', {
                'user': user,
                'error': f"Error processing enhanced forecasting: {e}",
                'forecast_price': [],
                'forecast_demand': [],
                'forecast_summary': {
                    '7_day_avg_price': 0,
                    '30_day_avg_price': 0,
                    '7_day_total_demand': 0,
                    '30_day_total_demand': 0,
                    'next_day_price': 0,
                    'next_day_demand': 0,
                },
                'insights': "Unable to generate enhanced insights due to processing error.",
                'summary_stats': {}
            })
    
    def format_forecasts_for_template(self, forecasts):
        """Format forecast data for Django template"""
        formatted = {}
        
        for period in ['7_day', '30_day']:
            if period not in forecasts:
                formatted[period] = {'price': [], 'demand': []}
                continue
                
            
            price_formatted = []
            for _, row in forecasts[period]['price'].iterrows():
                price_formatted.append({
                    'ds': row['ds'].strftime('%Y-%m-%d'),
                    'day_name': row['ds'].strftime('%A'),
                    'yhat': round(row['yhat'], 2),
                    'yhat_lower': round(row['yhat_lower'], 2),
                    'yhat_upper': round(row['yhat_upper'], 2),
                    'confidence': round(((row['yhat_upper'] - row['yhat_lower']) / row['yhat']) * 100, 1)
                })
            
            
            demand_formatted = []
            for _, row in forecasts[period]['demand'].iterrows():
                demand_formatted.append({
                    'ds': row['ds'].strftime('%Y-%m-%d'),
                    'day_name': row['ds'].strftime('%A'),
                    'yhat': round(row['yhat'], 0),
                    'yhat_lower': round(row['yhat_lower'], 0),
                    'yhat_upper': round(row['yhat_upper'], 0),
                    'confidence': round(((row['yhat_upper'] - row['yhat_lower']) / row['yhat']) * 100, 1) if row['yhat'] > 0 else 0
                })
            
            formatted[period] = {
                'price': price_formatted,
                'demand': demand_formatted
            }
        
        return formatted
    
    def format_for_existing_template(self, forecast_7_day):
        """Format 7-day forecast data for your existing template structure"""
        
        if not forecast_7_day or 'price' not in forecast_7_day or 'demand' not in forecast_7_day:
            return [], []
        
        price_data = forecast_7_day['price']
        demand_data = forecast_7_day['demand']
        
        
        forecast_price_formatted = []
        for _, row in price_data.iterrows():
            forecast_price_formatted.append({
                'ds': row['ds'].strftime('%Y-%m-%d'),
                'yhat': round(row['yhat'], 2),
                'yhat_lower': round(row['yhat_lower'], 2),
                'yhat_upper': round(row['yhat_upper'], 2)
            })
        
        
        forecast_demand_formatted = []
        for _, row in demand_data.iterrows():
            forecast_demand_formatted.append({
                'ds': row['ds'].strftime('%Y-%m-%d'),
                'yhat': round(row['yhat'], 0),
                'yhat_lower': round(row['yhat_lower'], 0),
                'yhat_upper': round(row['yhat_upper'], 0)
            })
        
        return forecast_price_formatted, forecast_demand_formatted
    
    def calculate_forecast_summary(self, forecasts):
        """Calculate forecast summary data for your frontend cards"""
        
        summary = {
            '7_day_avg_price': 0,
            '30_day_avg_price': 0,
            '7_day_total_demand': 0,
            '30_day_total_demand': 0,
            'next_day_price': 0,
            'next_day_demand': 0,
        }
        
        try:
            
            if '7_day' in forecasts and 'price' in forecasts['7_day'] and 'demand' in forecasts['7_day']:
                price_7_day = forecasts['7_day']['price']
                demand_7_day = forecasts['7_day']['demand']
                
                if len(price_7_day) > 0:
                    summary['7_day_avg_price'] = round(price_7_day['yhat'].mean(), 2)
                    summary['next_day_price'] = round(price_7_day.iloc[0]['yhat'], 2)
                
                if len(demand_7_day) > 0:
                    summary['7_day_total_demand'] = int(demand_7_day['yhat'].sum())
                    summary['next_day_demand'] = int(demand_7_day.iloc[0]['yhat'])
            
            
            if '30_day' in forecasts and 'price' in forecasts['30_day'] and 'demand' in forecasts['30_day']:
                price_30_day = forecasts['30_day']['price']
                demand_30_day = forecasts['30_day']['demand']
                
                if len(price_30_day) > 0:
                    summary['30_day_avg_price'] = round(price_30_day['yhat'].mean(), 2)
                
                if len(demand_30_day) > 0:
                    summary['30_day_total_demand'] = int(demand_30_day['yhat'].sum())
            
        except Exception as e:
            print(f"Error calculating forecast summary: {e}")
        
        return summary
    
    def calculate_detailed_summary_statistics(self, forecasts, historical_df):
        """Calculate comprehensive summary statistics (keeping existing method)"""
        try:
            stats = {}
            
            for period in ['7_day', '30_day']:
                if period not in forecasts:
                    continue
                    
                price_data = forecasts[period].get('price', pd.DataFrame())
                demand_data = forecasts[period].get('demand', pd.DataFrame())
                
                if len(price_data) == 0 or len(demand_data) == 0:
                    continue
                
                
                avg_price = price_data['yhat'].mean()
                min_price = price_data['yhat'].min()
                max_price = price_data['yhat'].max()
                price_volatility = price_data['yhat'].std()
                
                
                total_demand = demand_data['yhat'].sum()
                avg_daily_demand = demand_data['yhat'].mean()
                demand_volatility = demand_data['yhat'].std()
                
                
                historical_avg_price = historical_df['average_price'].mean()
                historical_avg_demand = historical_df['total_quantity_sold'].mean()
                
                price_change = ((avg_price - historical_avg_price) / historical_avg_price) * 100
                demand_change = ((avg_daily_demand - historical_avg_demand) / historical_avg_demand) * 100
                
                
                best_price_idx = price_data['yhat'].idxmax()
                worst_price_idx = price_data['yhat'].idxmin()
                
                best_day = {
                    'date': price_data.iloc[best_price_idx]['ds'].strftime('%A, %B %d'),
                    'price': round(price_data.iloc[best_price_idx]['yhat'], 2)
                }
                
                worst_day = {
                    'date': price_data.iloc[worst_price_idx]['ds'].strftime('%A, %B %d'),
                    'price': round(price_data.iloc[worst_price_idx]['yhat'], 2)
                }
                
                
                if price_change > 5:
                    sentiment = "Bullish"
                elif price_change < -5:
                    sentiment = "Bearish"
                else:
                    sentiment = "Stable"
                
                
                if price_volatility > avg_price * 0.15:
                    risk_level = "High"
                elif price_volatility > avg_price * 0.08:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                
                stats[period] = {
                    'avg_price': round(avg_price, 2),
                    'price_range': f"R{min_price:.2f} - R{max_price:.2f}",
                    'price_volatility': round(price_volatility, 2),
                    'total_demand': int(total_demand),
                    'avg_daily_demand': int(avg_daily_demand),
                    'demand_volatility': round(demand_volatility, 0),
                    'price_change_pct': round(price_change, 1),
                    'demand_change_pct': round(demand_change, 1),
                    'best_day': best_day,
                    'worst_day': worst_day,
                    'market_sentiment': sentiment,
                    'risk_level': risk_level,
                    'revenue_potential': int(total_demand * avg_price)
                }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating detailed summary statistics: {e}")
            return {}
        
    def generate_enhanced_insights(self, forecasts, historical_df):
        """Generate comprehensive market insights"""
        try:
            insights = []
            
            
            if '7_day' in forecasts and '30_day' in forecasts:
                
                
                price_7_day = forecasts['7_day'].get('price', pd.DataFrame())
                demand_7_day = forecasts['7_day'].get('demand', pd.DataFrame())
                price_30_day = forecasts['30_day'].get('price', pd.DataFrame())
                demand_30_day = forecasts['30_day'].get('demand', pd.DataFrame())
                
                if len(price_7_day) == 0 or len(price_30_day) == 0:
                    insights.append("⚠️ Insufficient forecast data for detailed analysis.")
                    return "\n".join(insights)
                
                
                week_avg = price_7_day['yhat'].mean()
                month_avg = price_30_day['yhat'].mean()
                historical_avg = historical_df['average_price'].mean()
                
                if week_avg > month_avg:
                    insights.append("📈 **Short-term Opportunity**: Prices are expected to be higher in the next week compared to the monthly average. Consider selling sooner rather than later.")
                else:
                    insights.append("📊 **Market Patience**: Prices may improve over the month. Consider timing your sales strategically.")
                
                
                price_change = ((week_avg - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
                if price_change > 10:
                    insights.append(f"🚀 **Strong Market**: Prices are forecasted to be {price_change:.1f}% higher than recent averages - excellent selling conditions ahead.")
                elif price_change > 5:
                    insights.append(f"💰 **Good Market**: Prices are expected to be {price_change:.1f}% above recent levels - favorable conditions for sales.")
                elif price_change < -5:
                    insights.append(f"⚠️ **Challenging Market**: Prices may be {abs(price_change):.1f}% below recent levels - consider cost management strategies.")
                
                
                if len(price_7_day) > 0:
                    price_7_day_reset = price_7_day.reset_index(drop=True)
                    best_day_idx = price_7_day_reset['yhat'].idxmax()
                    best_day = price_7_day_reset.iloc[best_day_idx]['ds']
                    best_price = price_7_day_reset.iloc[best_day_idx]['yhat']
                    
                    insights.append(f"📅 **Best Selling Day**: {best_day.strftime('%A, %B %d')} - Expected price: R{best_price:.2f}/kg")
                
                
                if len(demand_7_day) > 0 and len(demand_30_day) > 0:
                    week_demand = demand_7_day['yhat'].sum()
                    month_demand = demand_30_day['yhat'].sum()
                    
                    if week_demand > month_demand / 4:
                        insights.append("📦 **Strong Initial Demand**: Higher demand expected in the first week - good opportunity for bulk sales.")
                
                
                if len(price_7_day) > 0 and len(demand_7_day) > 0:
                    week_revenue = demand_7_day['yhat'].sum() * week_avg
                    insights.append(f"💵 **7-Day Revenue Potential**: R{week_revenue:,.0f} based on forecasted demand and prices")
                
                insights.append("\n**📋 Recommended Actions:**")
                insights.append("• Monitor market conditions daily for optimal selling opportunities")
                insights.append("• Consider storage costs vs. potential price improvements")
                insights.append("• Plan logistics around peak demand days")
                
                
                current_month = datetime.now().month
                if current_month in [2, 3, 4]:  
                    insights.append("• **Storage Strategy**: Consider short-term storage as post-harvest prices may recover")
                elif current_month in [9, 10, 11]:  
                    insights.append("• **Market Timing**: Premium pricing period - maximize sales volume")
            
            else:
                insights.append("⚠️ Unable to generate detailed insights due to insufficient forecast data.")
            
            return "\n".join(insights)
            
        except Exception as e:
            print(f"Error generating enhanced insights: {e}")
            import traceback
            traceback.print_exc()
            return f"Market analysis in progress. Please check back in a few minutes for detailed insights."
        
    def calculate_summary_statistics(self, forecasts, historical_df):
        """Calculate comprehensive summary statistics"""
        try:
            stats = {}
            
            for period in ['7_day', '30_day']:
                if period not in forecasts:
                    continue
                    
                price_data = forecasts[period]['price']
                demand_data = forecasts[period]['demand']
                
                if len(price_data) == 0 or len(demand_data) == 0:
                    continue
                
                
                avg_price = price_data['yhat'].mean()
                min_price = price_data['yhat'].min()
                max_price = price_data['yhat'].max()
                price_volatility = price_data['yhat'].std()
                
                
                total_demand = demand_data['yhat'].sum()
                avg_daily_demand = demand_data['yhat'].mean()
                demand_volatility = demand_data['yhat'].std()
                
                
                historical_avg_price = historical_df['average_price'].mean()
                historical_avg_demand = historical_df['total_quantity_sold'].mean()
                
                price_change = ((avg_price - historical_avg_price) / historical_avg_price) * 100
                demand_change = ((avg_daily_demand - historical_avg_demand) / historical_avg_demand) * 100
                
                
                best_price_idx = price_data['yhat'].idxmax()
                worst_price_idx = price_data['yhat'].idxmin()
                
                best_day = {
                    'date': price_data.iloc[best_price_idx]['ds'].strftime('%A, %B %d'),
                    'price': round(price_data.iloc[best_price_idx]['yhat'], 2)
                }
                
                worst_day = {
                    'date': price_data.iloc[worst_price_idx]['ds'].strftime('%A, %B %d'),
                    'price': round(price_data.iloc[worst_price_idx]['yhat'], 2)
                }
                
                
                if price_change > 5:
                    sentiment = "Bullish"
                elif price_change < -5:
                    sentiment = "Bearish"
                else:
                    sentiment = "Stable"
                
                
                if price_volatility > avg_price * 0.15:
                    risk_level = "High"
                elif price_volatility > avg_price * 0.08:
                    risk_level = "Medium"
                else:
                    risk_level = "Low"
                
                stats[period] = {
                    'avg_price': round(avg_price, 2),
                    'price_range': f"R{min_price:.2f} - R{max_price:.2f}",
                    'price_volatility': round(price_volatility, 2),
                    'total_demand': int(total_demand),
                    'avg_daily_demand': int(avg_daily_demand),
                    'demand_volatility': round(demand_volatility, 0),
                    'price_change_pct': round(price_change, 1),
                    'demand_change_pct': round(demand_change, 1),
                    'best_day': best_day,
                    'worst_day': worst_day,
                    'market_sentiment': sentiment,
                    'risk_level': risk_level,
                    'revenue_potential': int(total_demand * avg_price)
                }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating summary statistics: {e}")
            return {}
    




def get_sa_market_calendar():
    """Get South African agricultural market calendar events"""
    current_year = datetime.now().year
    
    market_events = {
        f"{current_year}-02-15": "Peak harvest season begins",
        f"{current_year}-04-30": "Main harvest season ends", 
        f"{current_year}-05-15": "Storage season optimal period",
        f"{current_year}-09-01": "Pre-planting market activity increases",
        f"{current_year}-10-15": "New season supply concerns begin",
        f"{current_year}-12-15": "Holiday demand peak"
    }
    
    return market_events

def calculate_carbon_footprint_impact(fuel_price, distance_km=100):
    """Calculate estimated carbon footprint based on transport costs"""
    
    
    fuel_efficiency = 35
    
    
    co2_per_liter = 2.68
    
    
    fuel_needed = (distance_km / 100) * fuel_efficiency
    fuel_cost = fuel_needed * fuel_price
    co2_emissions = fuel_needed * co2_per_liter
    
    return {
        'fuel_cost': fuel_cost,
        'co2_emissions': co2_emissions,
        'cost_per_km': fuel_cost / distance_km
    }


def assess_weather_impact(temperature, rainfall, humidity):
    """Assess weather impact on potato storage and transport"""
    
    impact_score = 0
    impacts = []
    
    
    if temperature > 30:
        impact_score += 3
        impacts.append("High temperature increases storage costs")
    elif temperature < 5:
        impact_score += 2
        impacts.append("Low temperature may affect transport")
    
    
    if rainfall > 20:
        impact_score += 3
        impacts.append("Heavy rainfall may disrupt logistics")
    elif rainfall > 10:
        impact_score += 1
        impacts.append("Moderate rainfall may cause minor delays")
    
    
    if humidity > 85:
        impact_score += 2
        impacts.append("High humidity increases spoilage risk")
    
    
    if impact_score >= 6:
        severity = "High"
    elif impact_score >= 3:
        severity = "Medium"
    else:
        severity = "Low"
    
    return {
        'severity': severity,
        'impact_score': impact_score,
        'specific_impacts': impacts
    }