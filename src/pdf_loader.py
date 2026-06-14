import fitz

class PDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_text(self):
        document = fitz.open(self.file_path)
        full_text = ""
        for page in document:
            page_text = page.get_text()
            full_text += page_text + "\n"
        
        document.close()
        return full_text