from django.urls import path
from .views import UserRegistrationView, UserLoginView, LogoutView, UserProfileView
from . import views

urlpatterns = [
    path("register/", UserRegistrationView.as_view(), name="api_register"),
    path("login/", UserLoginView.as_view(), name="api_login"),
    path('profile/', UserProfileView.as_view(), name='user-profile'),
    path("logout/", LogoutView.as_view(), name="api_logout"),
    path('profile/', views.UserProfileView.as_view(), name='user-profile'),
    path('change-password/', views.change_password, name='change-password'),
    path('delete-account/', views.delete_account, name='delete-account'),
    path('forgot-password/', views.ForgotPasswordView.as_view(), name='forgot-password'),
]