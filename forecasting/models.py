# models.py
from django.db import models
from django.core.validators import FileExtensionValidator

class ForecastingModel(models.Model):
    PROVINCE_CHOICES = [
        ('KZN', 'KwaZulu-Natal'),
        ('GP', 'Gauteng'),
        ('WC', 'Western Cape'),
        ('EC', 'Eastern Cape'),
        ('NC', 'Northern Cape'),
        ('FS', 'Free State'),
        ('MP', 'Mpumalanga'),
        ('LP', 'Limpopo'),
        ('NW', 'North West'),
    ]
    
    crop_name = models.CharField(max_length=100)
    province = models.CharField(max_length=3, choices=PROVINCE_CHOICES)
    dataset = models.FileField(
        upload_to="datasets/",
        validators=[FileExtensionValidator(allowed_extensions=['csv'])]
    )
    
    # Model status - FIXED: Default to False
    is_initialized = models.BooleanField(default=False)
    last_forecast_date = models.DateTimeField(null=True, blank=True)
    
    # Training metrics
    price_model_accuracy = models.FloatField(null=True, blank=True, default=0.0)
    demand_model_accuracy = models.FloatField(null=True, blank=True, default=0.0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['crop_name', 'province']
    
    def __str__(self):
        return f"{self.crop_name} - {self.province}"
    
    
    
    @property
    def has_forecasts(self):
        """Check if model has any forecasts"""
        return self.forecasts.exists()


class ForecastResult(models.Model):
    model = models.ForeignKey(ForecastingModel, on_delete=models.CASCADE, related_name='forecasts')
    forecast_date = models.DateField()
    forecast_horizon = models.IntegerField()  # 7 or 30
    
    # Price forecast
    predicted_price = models.DecimalField(max_digits=10, decimal_places=2)
    price_lower_bound = models.DecimalField(max_digits=10, decimal_places=2)
    price_upper_bound = models.DecimalField(max_digits=10, decimal_places=2)
    
    # Demand forecast
    predicted_demand = models.DecimalField(max_digits=12, decimal_places=2)
    demand_lower_bound = models.DecimalField(max_digits=12, decimal_places=2)
    demand_upper_bound = models.DecimalField(max_digits=12, decimal_places=2)
    
    # Revenue estimate
    revenue_estimate = models.DecimalField(max_digits=15, decimal_places=2)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['forecast_date']
        unique_together = ['model', 'forecast_date', 'forecast_horizon']
        indexes = [
            models.Index(fields=['model', 'forecast_horizon', 'forecast_date']),
        ]
    
    def __str__(self):
        return f"{self.model.crop_name} - {self.forecast_date} ({self.forecast_horizon} days)"


class ExternalFactors(models.Model):
    """Store external factors for South African context"""
    date = models.DateField()
    province = models.CharField(max_length=3)
    
    # Weather data
    avg_temperature = models.FloatField(null=True, blank=True)
    rainfall_mm = models.FloatField(null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True)
    
    # Economic factors
    fuel_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    exchange_rate = models.DecimalField(max_digits=10, decimal_places=4, null=True, blank=True)
    inflation_rate = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    
    # Seasonal indicators
    is_holiday = models.BooleanField(default=False)
    is_harvest_season = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-date']
        unique_together = ['date', 'province']
    
    def __str__(self):
        return f"{self.date} - {self.province}"