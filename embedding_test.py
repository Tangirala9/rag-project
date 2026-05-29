from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

sentences = [
    "Python Programming",
    "Machine Learning",
    "Data Analysis"
]

embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)

print("\nFirst Vector:\n")

print(embeddings[0][:10])