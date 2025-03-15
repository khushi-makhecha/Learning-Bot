from django.contrib import admin
from django.urls import path, include
from .views import create_index

urlpatterns = [
    path("create_index/", create_index, name="create_index"),
]
