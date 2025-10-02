from django.contrib import admin

from .models import ForecastingModel, ForecastResult

@admin.register(ForecastingModel)
class ForecastingModelAdmin(admin.ModelAdmin):
    list_display = ('crop_name', 'province', 'is_initialized', 'last_forecast_date', 'created_at')
    list_filter = ('province', 'is_initialized', 'created_at')
    search_fields = ('crop_name', 'province')

@admin.register(ForecastResult)
class ForecastResultAdmin(admin.ModelAdmin):
    list_display = ('model', 'forecast_date', 'forecast_horizon', 'predicted_price', 'predicted_demand')
    list_filter = ('forecast_horizon', 'forecast_date')
    search_fields = ('model__crop_name', 'model__province')
