import fitz  # PyMuPDF
from pathlib import Path

# Input and output paths
pdf_path = Path("../data/imirire.pdf")  # Change to match your file name
txt_path = Path("../data/kinyarwanda.txt")

# Open PDF
doc = fitz.open(pdf_path)

# Extract text from all pages
full_text = ""
for page in doc:
    full_text += page.get_text() + "\n"

# Save as UTF-8 text
txt_path.write_text(full_text.strip(), encoding="utf-8")

print(f"✅ Extracted text saved to {txt_path}")
