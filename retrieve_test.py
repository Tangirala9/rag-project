from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Load ChromaDB
client = chromadb.PersistentClient(
    path="vector_store"
)

collection = client.get_collection(
    name="documents"
)

# User Question
query = input("Ask Question: ")

# Query embedding
query_embedding = model.encode(query)

# Search
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=2
)

print("\nRetrieved Chunks:\n")

for doc in results["documents"][0]:
    print(doc)
    print("\n" + "="*50 + "\n")