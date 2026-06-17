from src.pdf_loader import PDFLoader
from src.chunker import TextChunker
from src.embedder import Embedder
from src.search import SemanticSearcher
from src.vector_store import VectorStore
from src.pipeline import IndexingPipeline

def main():

    # Initialize the indexing pipeline
    pdf_directory = "data/pdfs/"
    pipeline = IndexingPipeline(pdf_directory=pdf_directory)
    (chunks, embeddings, embedder, vector_store) = pipeline.build()

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
    vector_store_results = vector_store.search(query_embedding=query_embedding, top_k=3, document_name="DeepLearningforDummies.pdf")
    for result in vector_store_results:

        print(f"Distance: {result['distance']:.4f}")  
        print(f"Document: {result['document_name']} - Page: {result['page_number']} - Chunk ID: {result['chunk_id']}")  
        print(result["chunk"][:300])  # Print first 300 characters of the chunk
        print("-" * 50)

if __name__ == "__main__":    
    main()