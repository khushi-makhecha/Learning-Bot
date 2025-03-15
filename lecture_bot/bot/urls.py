from django.contrib import admin
from django.urls import path, include
from .views import create_index, create_llama_index

urlpatterns = [
    path("create_pc_db/", create_index, name="create_index"),
    path("create_llama_db/", create_llama_index, name="create_llama_index"),
]
