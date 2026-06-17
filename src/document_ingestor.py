import os
from src.pdf_loader import PDFLoader

class DocumentIngestor:

    def __init__(self,pdf_directory):
        self.pdf_directory = (pdf_directory)

    def load_documents(self):

        documents = []
        for filename in os.listdir(self.pdf_directory):

            if not filename.endswith(".pdf"):
                continue

            pdf_path = os.path.join(self.pdf_directory,filename)

            loader = PDFLoader(pdf_path)

            pages = (loader.extract_pages())

            documents.append({
                "document_name":filename,
                "pages":pages
            })

        return documents