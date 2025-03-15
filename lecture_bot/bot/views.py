from django.shortcuts import render
from .helpers.pinecone import create_pinecone_index
from .helpers.llama import create_llama_pc_index, create_llama_embeddings, query_llama
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Create your views here.
def create_index(request):
    index_created = create_pinecone_index()
    return HttpResponse("Index created", status=200) if index_created else HttpResponse("Failed to create index", status=500)


def create_llama_index(request):
    index_created = create_llama_pc_index()
    return HttpResponse("Index created", status=200) if index_created else HttpResponse("Failed to create index", status=500)


def create_llama_emb(request):
    embeddings_created = create_llama_embeddings()
    return HttpResponse("Embeddings created", status=200) if embeddings_created else HttpResponse("Failed to create embeddings", status=500)

@csrf_exempt
def post_query(request):
    data = json.loads(request.body)
    query = data.get("query")
    response = query_llama(query)
    return HttpResponse(response, status=200)