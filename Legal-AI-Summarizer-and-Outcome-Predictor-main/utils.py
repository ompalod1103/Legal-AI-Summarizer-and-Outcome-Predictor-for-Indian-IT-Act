from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF into a single string.
    """
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
    return text


def load_pdf_chunks(pdf_path: str, max_pages: int = 25) -> list:
    """
    Load PDF with page limit to conserve tokens
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to process (default: 25)
    
    Returns:
        List of text chunks
    """
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    total_pages = len(documents)
    # print(f"📄 PDF has {total_pages} pages, processing first {min(max_pages, total_pages)} pages")
    
    # Limit pages for demo efficiency
    documents = documents[:max_pages]
    
    # Smaller chunks = better for token limits
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # Smaller than before
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks]