from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf = PdfReader("data/AI_ML.pdf")

text = ""

for page in pdf.pages:
    text += page.extract_text()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("Total Chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0])