import os
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()


def create_llama_pc_index():
    try:
        # Set up Pinecone API key
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("LLAMA_PC_INDEX_NAME")
        dimension = int(os.getenv("PINECONE_DIMENSION"))
        cloud = os.getenv("PINECONE_CLOUD")
        region = os.getenv("PINECONE_REGION")

        # Create Pinecone Vector Store
        pc = Pinecone(api_key=api_key)

        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="dotproduct",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

        pinecone_index = pc.Index(index_name)

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
        )

        return vector_store
    except Exception as e:
        print(f"Error creating Pinecone index: {e}")
        return False
