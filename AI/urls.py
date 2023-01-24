from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('figdetector/', include('figdetector.urls')),
    path('wildfire/', include('wildfire.urls')),
    path('admin/', admin.site.urls),
]
