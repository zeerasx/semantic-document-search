from sentence_transformers import SentenceTransformer

class Embedder:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(
            model_name, token=False
        )

    def embed_text(self, text):
        return self.model.encode(text)

    def embed_chunks(self, chunks):

        texts = [
            chunk["chunk"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(texts)

        return embeddings