class TextChunker:

    def __init__(
        self,
        chunk_size=500,
        overlap=100
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):

        chunks = []

        start = 0
        chunk_id = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk_text = text[start:end]

            chunks.append({
                "chunk_id":chunk_id, 
                "chunk": chunk_text})
            
            chunk_id += 1

            start += (
                self.chunk_size
                - self.overlap
            )

            

        return chunks