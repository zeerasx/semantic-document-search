import fitz

class PDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_pages(self):
        document = fitz.open(self.pdf_path)

        pages = []
        for page_num in range(len(document)):

            page = document.load_page(page_num)
            text = page.get_text()
            pages.append({
                "page_number":page_num + 1,
                "text":text
            })
        return pages
    
    def extract_text(self):
        pages = self.extract_pages()
        return "\n".join(
            page["text"]
            for page in pages
        )