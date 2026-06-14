from src.pdf_loader import PDFLoader
from src.chunker import TextChunker

def main():
    # Ingestion
    pdf_path = "data/pdfs/DeepLearningforDummies.pdf"
    loader = PDFLoader(pdf_path)
    text = loader.extract_text()
    
    #Chunking
    chunker = TextChunker(
        chunk_size=500,
        overlap=100
    )

    chunks = chunker.chunk_text(text)

    print(
        f"Total Chunks: {len(chunks)}"
    )

    for chunk in chunks[:3]:
        print("======================================")
        print(chunk["chunk_id"])
        print("======================================")
        print(chunk["chunk"])


if __name__ == "__main__":    
    main()