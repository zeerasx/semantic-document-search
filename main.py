from src.pdf_loader import PDFLoader
from src.chunker import TextChunker
from src.embedder import Embedder
from src.search import SemanticSearcher
from src.vector_store import VectorStore

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

    ## Performing Manual search
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

    # Perform search using Vector Store
    print("///////////////////////////VECTOR STORE SEARCH//////////////////////////////////////////////////////////")
    vector_store = VectorStore()
    vector_store.add_chunks(chunks, embeddings)
    vector_store_results = vector_store.search(query_embedding=query_embedding, top_k=3)
    for doc, distance in zip(
        vector_store_results["documents"][0],
        vector_store_results["distances"][0]
    ):

        print("\n")
        print(f"Distance: {distance:.4f}")
        print(doc[:300])
        print("-" * 50)

if __name__ == "__main__":    
    main()