from django.shortcuts import render

# Create your views here.
from rest_framework import viewsets, status
from django.contrib.auth.decorators import login_required
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.db.models import Avg, Sum, Max, Min
from django.utils import timezone
from django.contrib.auth.decorators import user_passes_test
import pandas as pd
from .ai_insights import AIIntelligenceEngine
from django.utils import timezone
from .forecaster import LagLLAMAForecaster
from datetime import timedelta
from .models import ForecastingModel, ForecastResult
from .serializers import (
    ForecastingModelSerializer, 
    ForecastResultSerializer,
    ForecastSummarySerializer,
    DetailedForecastSerializer
)
from .tasks import initialize_forecast_model, generate_forecasts

import logging

logger = logging.getLogger(__name__)

def is_admin(user):
    return user.is_staff or user.is_superuser

@login_required
@user_passes_test(is_admin)
def forecast_initialize_page(request):
    """Render the forecast initialization page - Admin only"""
    return render(request, 'forecasting/initialize.html')

@login_required
def forecast_dashboard(request):
    """Dashboard to view all models with forecast data"""
    models = ForecastingModel.objects.all().order_by('-created_at')

    model_data = []
    for model in models:
        forecasts_7 = ForecastResult.objects.filter(model=model, forecast_horizon=7)
        forecasts_30 = ForecastResult.objects.filter(model=model, forecast_horizon=30)

        avg_price_7 = forecasts_7.aggregate(Avg('predicted_price'))['predicted_price__avg'] or 0
        total_demand_7 = forecasts_7.aggregate(Sum('predicted_demand'))['predicted_demand__sum'] or 0

        avg_price_30 = forecasts_30.aggregate(Avg('predicted_price'))['predicted_price__avg'] or 0
        total_demand_30 = forecasts_30.aggregate(Sum('predicted_demand'))['predicted_demand__sum'] or 0

        summary_7 = _calculate_summary_stats(forecasts_7)
        summary_30 = _calculate_summary_stats(forecasts_30)

        # Generate AI insights
        ai_engine = AIIntelligenceEngine()
        insights_text = ai_engine.generate_insights(model.id)
        
        # Pre-process insights for template display
        insights_lines = []
        for line in insights_text.split('\n'):
            line = line.strip()
            if line:  # Only process non-empty lines
                # Remove markdown-style bold markers (all asterisks)
                clean_line = line.replace("**", "")

                # Determine line type
                if line.startswith('**') and line.endswith('**'):
                    # This is a heading
                    insights_lines.append({
                        'text': clean_line,
                        'type': 'heading'
                    })
                elif line.startswith('•') or line.startswith('📈') or line.startswith('📊') or line.startswith('⚠️') or line.startswith('🚀') or line.startswith('📦') or line.startswith('💰') or line.startswith('🛡️') or line.startswith('🎯'):
                    # This is a bullet point or icon line
                    insights_lines.append({
                        'text': clean_line,
                        'type': 'bullet'
                    })
                else:
                    # Regular paragraph
                    insights_lines.append({
                        'text': clean_line,
                        'type': 'paragraph'
                    })


        model_data.append({
            'model': model,
            'crop_name': model.crop_name,
            'province': model.province,
            'forecast_summary': {
                '7_day_avg_price': round(float(avg_price_7), 2),
                '7_day_total_demand': round(float(total_demand_7), 2),
                '30_day_avg_price': round(float(avg_price_30), 2),
                '30_day_total_demand': round(float(total_demand_30), 2)
            },
            'summary_stats': {
                '7_day': summary_7,
                '30_day': summary_30
            },
            'data_summary': {
                'external_factors_used': True
            },
            'insights': insights_text,  # Keep original for backup
            'insights_lines': insights_lines,  # Pre-processed for template
            'has_forecasts': forecasts_7.exists() or forecasts_30.exists()
        })

    # recent one = first in ordered queryset
    recent_model = model_data[0] if model_data else None

    context = {
        'model_data': model_data,
        'recent_model': recent_model,
    }
    return render(request, 'forecasting/forecast-dashboard.html', context)



@login_required
def forecast_detail_page(request, pk):
    """View forecast results for a specific model"""
    model = ForecastingModel.objects.get(pk=pk)
    
    context = {
        'model': model
    }
    return render(request, 'forecasting/detail.html', context)


class ForecastingModelViewSet(viewsets.ModelViewSet):
    queryset = ForecastingModel.objects.all()
    serializer_class = ForecastingModelSerializer
    permission_classes = [IsAuthenticated]
    
    def get_permissions(self):
        if self.action in ['create', 'update', 'destroy']:
            return [IsAdminUser()]
        return [IsAuthenticated()]
    
    # views.py - Update the create method
# views.py - Make sure this is correct
    def create(self, request, *args, **kwargs):
        """Create and immediately generate forecasts"""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        model = serializer.save()
        
        # Run forecast IMMEDIATELY (not async)
        try:
            # Load dataset
            df = pd.read_csv(model.dataset.path)
            
            # Train model
            forecaster = LagLLAMAForecaster()
            forecaster.train(df)
            
            # Generate forecasts
            forecasts_7 = forecaster.forecast(horizon=7)
            forecasts_30 = forecaster.forecast(horizon=30)
            
            # Save to database
            for forecast in forecasts_7:
                ForecastResult.objects.create(
                    model=model,
                    forecast_date=forecast['date'],
                    forecast_horizon=7,
                    predicted_price=forecast['price_yhat'],
                    price_lower_bound=forecast['price_yhat_lower'],
                    price_upper_bound=forecast['price_yhat_upper'],
                    predicted_demand=forecast['demand_yhat'],
                    demand_lower_bound=forecast['demand_yhat_lower'],
                    demand_upper_bound=forecast['demand_yhat_upper'],
                    revenue_estimate=forecast['revenue_estimate']
                )
            
            for forecast in forecasts_30:
                ForecastResult.objects.create(
                    model=model,
                    forecast_date=forecast['date'],
                    forecast_horizon=30,
                    predicted_price=forecast['price_yhat'],
                    price_lower_bound=forecast['price_yhat_lower'],
                    price_upper_bound=forecast['price_yhat_upper'],
                    predicted_demand=forecast['demand_yhat'],
                    demand_lower_bound=forecast['demand_yhat_lower'],
                    demand_upper_bound=forecast['demand_yhat_upper'],
                    revenue_estimate=forecast['revenue_estimate']
                )
            
            # Update model
            model.is_initialized = True
            model.last_forecast_date = timezone.now()
            model.save()
            
            return Response({
                'status': 'success',
                'model_id': model.id,
                'forecasts_generated': len(forecasts_7) + len(forecasts_30)
            })
            
        except Exception as e:
            model.delete()  # Clean up if failed
            return Response({'error': str(e)}, status=400)
    
    @action(detail=True, methods=['get'])
    def forecast(self, request, pk=None):
        """Get forecast results for a specific model"""
        model = self.get_object()
        
        if not model.is_initialized:
            return Response({
                'error': 'Model not yet initialized. Please wait.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        horizon = int(request.query_params.get('horizon', 7))
        
        # Get forecast results
        forecasts = ForecastResult.objects.filter(
            model=model,
            forecast_horizon=horizon
        ).order_by('forecast_date')
        
        if not forecasts.exists():
            # Trigger forecast generation
            generate_forecasts.delay(model.id, horizon)
            return Response({
                'status': 'generating',
                'message': 'Forecast is being generated'
            }, status=status.HTTP_202_ACCEPTED)
        
        # Prepare response data
        forecast_price = []
        forecast_demand = []
        
        for f in forecasts:
            forecast_price.append({
                'ds': f.forecast_date.strftime('%Y-%m-%d'),
                'yhat': float(f.predicted_price),
                'yhat_lower': float(f.price_lower_bound),
                'yhat_upper': float(f.price_upper_bound)
            })
            
            forecast_demand.append({
                'ds': f.forecast_date.strftime('%Y-%m-%d'),
                'yhat': float(f.predicted_demand),
                'yhat_lower': float(f.demand_lower_bound),
                'yhat_upper': float(f.demand_upper_bound)
            })
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(forecasts)
        
        return Response({
            'crop_name': model.crop_name,
            'province': model.province,
            'forecast_price': forecast_price,
            'forecast_demand': forecast_demand,
            'forecast_summary': {
                f'{horizon}_day_avg_price': summary_stats['avg_price'],
                f'{horizon}_day_total_demand': summary_stats['total_demand']
            },
            'summary_stats': {
                f'{horizon}_day': summary_stats
            },
            'data_summary': {
                'external_factors_used': True
            }
        })
    
    @action(detail=True, methods=['get'])
    def combined_forecast(self, request, pk=None):
        """Get both 7-day and 30-day forecasts with actual averages"""
        model = self.get_object()
        
        if not model.is_initialized:
            return Response({
                'error': 'Model not yet initialized'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Get forecasts from database
        forecasts_7 = ForecastResult.objects.filter(
            model=model,
            forecast_horizon=7
        ).order_by('forecast_date')
        
        forecasts_30 = ForecastResult.objects.filter(
            model=model,
            forecast_horizon=30
        ).order_by('forecast_date')
        
        # Calculate ACTUAL averages from database
        avg_price_7 = forecasts_7.aggregate(Avg('predicted_price'))['predicted_price__avg'] or 0
        total_demand_7 = forecasts_7.aggregate(Sum('predicted_demand'))['predicted_demand__sum'] or 0
        
        avg_price_30 = forecasts_30.aggregate(Avg('predicted_price'))['predicted_price__avg'] or 0
        total_demand_30 = forecasts_30.aggregate(Sum('predicted_demand'))['predicted_demand__sum'] or 0
        
        response_data = {
            'crop_name': model.crop_name,
            'province': model.province,
            'forecast_price': self._format_forecasts(forecasts_7, 'price'),
            'forecast_demand': self._format_forecasts(forecasts_7, 'demand'),
            'forecast_30_day': {
                'price': self._format_forecasts(forecasts_30, 'price'),
                'demand': self._format_forecasts(forecasts_30, 'demand')
            },
            # ACTUAL AVERAGES FROM DATABASE
            'forecast_summary': {
                '7_day_avg_price': round(float(avg_price_7), 2),
                '7_day_total_demand': round(float(total_demand_7), 2),
                '30_day_avg_price': round(float(avg_price_30), 2),
                '30_day_total_demand': round(float(total_demand_30), 2)
            },
            'summary_stats': {
                '7_day': self._calculate_summary_stats(forecasts_7),
                '30_day': self._calculate_summary_stats(forecasts_30)
            },
            'data_summary': {
                'external_factors_used': True
            }
        }
        
        return Response(response_data)
    
    @action(detail=True, methods=['post'])
    def refresh(self, request, pk=None):
        """Manually trigger forecast refresh"""
        model = self.get_object()
        
        if not model.is_initialized:
            return Response({
                'error': 'Model must be initialized first'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Trigger forecast regeneration
        generate_forecasts.delay(model.id, 7)
        generate_forecasts.delay(model.id, 30)
        
        return Response({
            'status': 'refresh_initiated',
            'message': 'Forecasts are being regenerated'
        })
    
    def _format_forecasts(self, forecasts, forecast_type):
        """Format forecast results for response"""
        result = []
        for f in forecasts:
            if forecast_type == 'price':
                result.append({
                    'ds': f.forecast_date.strftime('%Y-%m-%d'),
                    'yhat': float(f.predicted_price),
                    'yhat_lower': float(f.price_lower_bound),
                    'yhat_upper': float(f.price_upper_bound)
                })
            else:  # demand
                result.append({
                    'ds': f.forecast_date.strftime('%Y-%m-%d'),
                    'yhat': float(f.predicted_demand),
                    'yhat_lower': float(f.demand_lower_bound),
                    'yhat_upper': float(f.demand_upper_bound)
                })
        return result
    
    def _calculate_avg(self, forecasts, field):
        """Calculate average for field"""
        if field == 'price':
            return forecasts.aggregate(Avg('predicted_price'))['predicted_price__avg'] or 0
        return forecasts.aggregate(Avg('predicted_demand'))['predicted_demand__avg'] or 0
    
    def _calculate_total(self, forecasts, field):
        """Calculate total for field"""
        if field == 'demand':
            return forecasts.aggregate(Sum('predicted_demand'))['predicted_demand__sum'] or 0
        return 0
    
def _calculate_summary_stats(forecasts):
    """Calculate summary statistics for forecasts"""
    if not forecasts.exists():
        return {}
    
    # Price statistics
    avg_price = forecasts.aggregate(Avg('predicted_price'))['predicted_price__avg'] or 0
    max_price = forecasts.aggregate(Max('predicted_price'))['predicted_price__max'] or 0
    min_price = forecasts.aggregate(Min('predicted_price'))['predicted_price__min'] or 0
    
    # Demand statistics
    total_demand = forecasts.aggregate(Sum('predicted_demand'))['predicted_demand__sum'] or 0
    
    # Revenue potential
    revenue = sum(float(f.revenue_estimate) for f in forecasts)
    
    # Market sentiment
    if forecasts.count() > 1:
        first_price = float(forecasts.first().predicted_price)
        last_price = float(forecasts.last().predicted_price)
        price_change = ((last_price - first_price) / first_price) * 100 if first_price else 0
        
        if price_change > 5:
            sentiment = 'Bullish'
        elif price_change < -5:
            sentiment = 'Bearish'
        else:
            sentiment = 'Neutral'
    else:
        sentiment = 'Neutral'
        price_change = 0
    
    # Risk level based on price range (removed StdDev)
    volatility = (float(max_price) - float(min_price)) / float(avg_price) if avg_price else 0
    
    if volatility < 0.1:
        risk = 'Low'
    elif volatility < 0.2:
        risk = 'Medium'
    else:
        risk = 'High'
    
    # Best selling day
    best_day_forecast = forecasts.order_by('-predicted_price').first()
    
    return {
        'avg_price': round(float(avg_price), 2),
        'total_demand': round(float(total_demand), 2),
        'revenue_potential': round(float(revenue), 2),
        'price_range': f"R{round(float(min_price), 2)} - R{round(float(max_price), 2)}",
        'market_sentiment': sentiment,
        'risk_level': risk,
        'price_change_pct': round(price_change, 2),
        'best_day': {
            'date': best_day_forecast.forecast_date.strftime('%Y-%m-%d'),
            'price': round(float(best_day_forecast.predicted_price), 2)
        } if best_day_forecast else None
    }
