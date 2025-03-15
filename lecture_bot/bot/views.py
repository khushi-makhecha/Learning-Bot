from django.shortcuts import render
from .helpers.pinecone import create_pinecone_index
from django.http import HttpResponse

# Create your views here.
def create_index(request):
    index_created = create_pinecone_index()
    return HttpResponse("Index created", status=200) if index_created else HttpResponse("Failed to create index", status=500)
