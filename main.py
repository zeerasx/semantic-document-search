from src.pdf_loader import PDFLoader

def main():
    pdf_path = "data/pdfs/DeepLearningforDummies.pdf"
    loader = PDFLoader(pdf_path)
    text = loader.extract_text()
    print(text[:3000])


if __name__ == "__main__":    
    main()