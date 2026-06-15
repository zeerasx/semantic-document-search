from src.pdf_loader import PDFLoader
from src.chunker import TextChunker
from src.embedder import Embedder
from src.search import SemanticSearcher

def main():
    # Ingestion
    pdf_path = "data/pdfs/DeepLearningforDummies.pdf"
    loader = PDFLoader(pdf_path)
    text = loader.extract_text()
    
    #Chunking
    print("////////////////////////////CHUNKING/////////////////////////////////////////////////////////")
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

    # Embedding
    print("///////////////////////////EMBEDDING//////////////////////////////////////////////////////////")
    embedder = Embedder()
    print("Embedding Chunks...")
    embeddings = embedder.embed_chunks(chunks)
    print("Embeddings Generated.")
    print(f"Total Embeddings: {len(embeddings)}")
    print(f"Embedding Dimension: {embeddings[0].shape}")
    print("Sample Embedding:")
    print(embeddings[0][:5])

    # Semantic Search

    ## Making query and its embedding
    query = input( "Enter search query: ")
    query_embedding = embedder.embed_text(query)

    ## Performing search
    searcher = SemanticSearcher()
    results = searcher.search(
        query_embedding=query_embedding,
        chunk_embeddings=embeddings,
        chunks=chunks,
        top_k=3
    )

    print("\nTop Results\n")
    for result in results:
        print("======================================")
        print(f"Score: {result['score']:.4f}")
        print("======================================")
        print(result["chunk"]["chunk"][:300])  # Print first 300 characters of the chunk

if __name__ == "__main__":    
    main()