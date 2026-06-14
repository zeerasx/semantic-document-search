from src.embedder import Embedder


def test_embedding_generation():

    embedder = Embedder()

    vector = embedder.embed_text(
        "Hello world"
    )

    assert len(vector) == 384