import fitz  # PyMuPDF
import camelot

def load_pdf(file_path):
    text_data = []

    # Extract text
    doc = fitz.open(file_path)
    for page in doc:
        text_data.append(page.get_text())

    # Extract tables
    tables = camelot.read_pdf(file_path, pages="all")
    table_data = [t.df.to_string() for t in tables]

    return text_data + table_data