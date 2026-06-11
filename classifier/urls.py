from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='index'),
    path('api/predict/', views.api_predict, name='api_predict'),
]