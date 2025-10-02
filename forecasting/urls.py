from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    ForecastingModelViewSet,
    forecast_initialize_page,
    forecast_dashboard,
    forecast_detail_page
)

router = DefaultRouter()
router.register(r'forecasting', ForecastingModelViewSet, basename='forecasting')

urlpatterns = [
    # API endpoints
    path('api/', include(router.urls)),
    
    # HTML page endpoints
    path('forecast/initiate/', forecast_initialize_page, name='forecast-initialize'),
    path('forecast/dashboard/', forecast_dashboard, name='forecast-dashboard'),
    path('forecast/<int:pk>/', forecast_detail_page, name='forecast-detail'),
]