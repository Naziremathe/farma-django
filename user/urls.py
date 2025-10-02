from django.urls import path
from .views import SignUpView, LoginView, EnhancedDashboardView, LogoutView, CropListView, crop_price_prediction_api, EnhancedCropUploadView, UserProfileView
from . import views

urlpatterns = [
    path("", SignUpView.as_view(), name="signup"),
    path("login/", LoginView.as_view(), name="login"),
    path("dashboard/", EnhancedDashboardView.as_view(), name="dashboard"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path('profile/', UserProfileView.as_view(), name='user-profile'),
    path('profile/', views.UserProfileView.as_view(), name='user-profile'),
    
    path('change-password/', views.change_password, name='change-password'),
    path('delete-account/', views.delete_account, name='delete-account'),



    path('crop-price-prediction/', crop_price_prediction_api, name='crop_price_prediction_api'),
    path('crops/upload/', EnhancedCropUploadView.as_view(), name='crop_upload'),
    path('list/', CropListView.as_view(), name='crop_list'),
]