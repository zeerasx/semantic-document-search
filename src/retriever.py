class Retriever:

    def __init__(
        self,
        embedder,
        vector_store
    ):

        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        query,
        top_k=5,
        document_name=None
    ):

        query_embedding = (
            self.embedder.embed_text(query)
        )

        return self.vector_store.search(
            query_embedding,
            top_k,
            document_name
        )