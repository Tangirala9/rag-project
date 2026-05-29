from pypdf import PdfReader

pdf = PdfReader("data/AI_ML.pdf")

print("Total Pages:", len(pdf.pages))

text = pdf.pages[0].extract_text()

print(text[:1000])