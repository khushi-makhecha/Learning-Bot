from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

def create_pinecone_index():
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME")
        dimension = int(os.getenv("PINECONE_DIMENSION"))
        cloud = os.getenv("PINECONE_CLOUD")
        region = os.getenv("PINECONE_REGION")

        # Set up Pinecone environment
        pc = Pinecone(api_key=api_key)

        # Create or connect to a vector index
        pc.create_index(name=index_name, dimension=dimension, spec=ServerlessSpec(cloud=cloud, region=region))  # Llama-based embeddings
        return True
    except Exception as e:
        print(f"Exception while creating Pinecone index: {e}")
        return False