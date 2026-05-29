from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Load PDF
pdf = PdfReader("data/AI_ML.pdf")

text = ""

for page in pdf.pages:
    text += page.extract_text()

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

# Chroma Client
client = chromadb.PersistentClient(
    path="vector_store"
)

collection = client.get_or_create_collection(
    name="documents"
)

# Store in ChromaDB
for i, chunk in enumerate(chunks):
    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embeddings[i].tolist()]
    )

print("Saved to ChromaDB Successfully!")