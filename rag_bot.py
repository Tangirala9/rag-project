from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
# Gemini API Key


genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model_gemini = genai.GenerativeModel(
    "gemini-2.0-flash"

)

# Embedding Model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ChromaDB
client = chromadb.PersistentClient(
    path="vector_store"
)

collection = client.get_collection(
    name="documents"
)

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=2
    )

    context = "\n".join(
        results["documents"][0]
    )

    prompt = f"""
Answer ONLY from the context below.

If answer is not present,
say:
'I could not find the answer in the documents.'

Context:
{context}

Question:
{query}
"""
    #
    # response = model_gemini.generate_content(
    #     prompt
    # )
    #
    # print("\nAnswer:\n")
    # print(response.text)
    print("\nRetrieved Context:\n")

    for doc in results["documents"][0]:
        print(doc)
        print("\n" + "=" * 50)

    print("\nSources:")
    print("AI_ML.pdf")