import os
import pandas as pd
import fitz  # PyMuPDF
# from pymupdf import fitz

from PIL import Image
import pytesseract
import io

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# === Function 1: Load Scraped Data from CSV ===
def load_scraped_data(csv_path):
    """
    Load scraped product data from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing scraped data.
    
    Returns:
        pandas.DataFrame: Loaded data.
    """
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found at: {csv_path}")
        return pd.DataFrame()
    
    try:
        data = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(data)} rows from {csv_path}")
        return data
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return pd.DataFrame()

# === Function 2: Extract Text from a Single PDF File ===
def extract_text_from_pdf(pdf_path):
    """
    Extract text from a single PDF file using PyMuPDF.
    If no text is detected, use OCR with pytesseract.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ''

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()

            if page_text.strip():
                # Use extracted text if available
                text += page_text
            else:
                # Use OCR if no text is found
                img = page.get_pixmap()
                img_bytes = img.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))
                ocr_text = pytesseract.image_to_string(image)
                text += ocr_text

        return text.strip()
    except Exception as e:
        print(f"❌ Error extracting from PDF {pdf_path}: {e}")
        return ""

# === Function 3: Load All PDFs from a Directory ===
def load_pdf_data_from_folder(folder_path):
    """
    Load text from multiple PDFs in a folder.
    
    Args:
        folder_path (str): Path to the folder containing PDF files.
    
    Returns:
        str: Combined text from all PDFs.
    """
    combined_text = ''
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found at: {folder_path}")
        return ""
    
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found in the folder.")
        return ""

    print(f"🔎 Found {len(pdf_files)} PDFs in '{folder_path}'")
    
    for file in pdf_files:
        pdf_path = os.path.join(folder_path, file)
        print(f"📄 Reading: {file}")
        extracted_text = extract_text_from_pdf(pdf_path)
        
        if extracted_text:
            combined_text += f"\n\n=== Extracted from {file} ===\n\n{extracted_text}"
    
    print("✅ PDF extraction complete.")
    return combined_text.strip()

# === Test Example ===
if __name__ == "__main__":
    # Test loading scraped data
    csv_path = "data/scraped_data.csv"
    scraped_data = load_scraped_data(csv_path)
    print("\nSample scraped data:")
    print(scraped_data.head())

    # Test loading PDF data
    folder_path = "data/pdfs"
    combined_text = load_pdf_data_from_folder(folder_path)
    if combined_text:
        print("\n✅ Combined PDF text length:", len(combined_text))
