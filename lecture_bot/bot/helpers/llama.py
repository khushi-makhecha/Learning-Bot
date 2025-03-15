from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

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
    

def get_pinecone_index():
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("LLAMA_PC_INDEX_NAME")

        pc = Pinecone(api_key=api_key)
        pinecone_index = pc.Index(index_name)

        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
        )

        return vector_store

    except Exception as e:
        print(f"Error getting Pinecone index: {e}")
        return None


def create_llama_embeddings():
    try:    
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize OpenAI embedding model
        embed_model = OpenAIEmbedding(
            api_key=openai_api_key,
            model="text-embedding-ada-002",  # This model outputs 1536 dimensions
            retry_on_error=True,
            max_retries=3
        )
        Settings.embed_model = embed_model
        
        # Load documents from directory
        documents = SimpleDirectoryReader("../lecture_bot/public/content").load_data()
        print(f"Loaded {len(documents)} documents")
        
        # Create vector store index
        vector_store = get_pinecone_index()  # Your existing Pinecone setup
        if not vector_store:
            raise ValueError("Failed to create Pinecone vector store")
        
        print("Starting to create index and embed documents...")

        # Use Pinecone storage explicitly
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            storage_context=storage_context,
            show_progress=True,
        )


        # print("Verifying index creation...")
        # query_engine = index.as_query_engine()
        # test_query = "test"  # Simple test query
        # response = query_engine.query(test_query)
        # print(f"Test query response received: {response}")
        print(f"Index successfully created and stored in Pinecone! {index}")
        return index
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return False
