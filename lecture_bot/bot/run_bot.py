from helpers.pinecone import create_pinecone_index

pinecone_index_created = create_pinecone_index()

print(f"Pinecone index created successfully") if pinecone_index_created else print("Failed to create Pinecone index")