from rest_framework import serializers
from .models import ForecastingModel, ForecastResult, ExternalFactors

class ForecastingModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastingModel
        fields = [
            'id', 'crop_name', 'province', 'dataset',
            'is_initialized', 'last_forecast_date',
            'price_model_accuracy', 'demand_model_accuracy',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'is_initialized', 'last_forecast_date',
            'price_model_accuracy', 'demand_model_accuracy',
            'created_at', 'updated_at'
        ]


class ForecastResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastResult
        fields = [
            'id', 'forecast_date', 'forecast_horizon',
            'predicted_price', 'price_lower_bound', 'price_upper_bound',
            'predicted_demand', 'demand_lower_bound', 'demand_upper_bound',
            'revenue_estimate', 'created_at'
        ]


class ForecastSummarySerializer(serializers.Serializer):
    """Summary statistics for frontend display"""
    horizon = serializers.IntegerField()
    avg_price = serializers.DecimalField(max_digits=10, decimal_places=2)
    total_demand = serializers.DecimalField(max_digits=12, decimal_places=2)
    revenue_potential = serializers.DecimalField(max_digits=15, decimal_places=2)
    price_range = serializers.CharField()
    market_sentiment = serializers.CharField()
    risk_level = serializers.CharField()
    price_change_pct = serializers.FloatField()
    best_day = serializers.DictField(required=False)


class DetailedForecastSerializer(serializers.Serializer):
    """Detailed forecast for tables"""
    ds = serializers.DateField()
    yhat = serializers.DecimalField(max_digits=10, decimal_places=2)
    yhat_lower = serializers.DecimalField(max_digits=10, decimal_places=2)
    yhat_upper = serializers.DecimalField(max_digits=10, decimal_places=2)