from src.pdf_loader import PDFLoader
from src.chunker import TextChunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.document_ingestor import DocumentIngestor

class IndexingPipeline:

    def __init__(
        self,
        pdf_path,
        chunk_size=500,
        overlap=100,
        model_name="all-MiniLM-L6-v2"
    ):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model_name = model_name

    def build(self):

        ingestor = DocumentIngestor(self.pdf_directory)

        documents = (ingestor.load_documents())

        all_chunks = []

        for document in documents:
            chunker = TextChunker(
                chunk_size=self.chunk_size,
                overlap=self.overlap
            )

            chunks = (chunker.chunk_pages(document["pages"],document["document_name"]))

            all_chunks.extend(chunks)
#/#
        embedder = Embedder(model_name=self.model_name)

        embeddings = (embedder.embed_chunks(all_chunks))

##
        vector_store = VectorStore()

        vector_store.reset()

        vector_store.add_chunks(
            all_chunks,
            embeddings
        )

        return (
            chunks,
            embeddings,
            embedder,
            vector_store
        )