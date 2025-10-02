from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),

    # Public HTML pages (mounted at root)
    path("", include("user.urls")),

    # API endpoints (separate namespace, e.g. /api/)
    path("", include("user.api_urls")),
    path("", include("forecasting.urls")),
]

if settings.DEBUG:  # only serve media in dev
    #urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
