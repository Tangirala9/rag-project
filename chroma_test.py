from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os

text = ""

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        print("Reading:", file)

        pdf = PdfReader(f"data/{file}")

        for page in pdf.pages:
            text += page.extract_text() + "\n"

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Chunks:", len(chunks))

# Embedding Model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(chunks)

print("Embeddings Created")

# ChromaDB
client = chromadb.PersistentClient(
    path="vector_store"
)

# Delete old collection if exists
try:
    client.delete_collection("documents")
except:
    pass
try:
    client.delete_collection("documents")
except:
    pass

collection = client.create_collection(
    name="documents"
)

# Store chunks
for i, chunk in enumerate(chunks):
    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embeddings[i].tolist()]
    )

print("Saved to ChromaDB Successfully!")