from src.chunker import TextChunker


def test_chunk_creation():

    text = "A" * 2000

    chunker = TextChunker(
        chunk_size=500,
        overlap=100
    )

    chunks = chunker.chunk_text(text)

    assert len(chunks) > 0