import numpy as np


class SemanticSearcher:

    def cosine_similarity(
        self,
        vector_a,
        vector_b
    ):

        numerator = np.dot(
            vector_a,
            vector_b
        )

        denominator = (
            np.linalg.norm(vector_a)
            *
            np.linalg.norm(vector_b)
        )

        return numerator / denominator
    

    def search(
        self,
        query_embedding,
        chunk_embeddings,
        chunks,
        top_k=3
    ):

        scores = []

        for chunk, embedding in zip(
            chunks,
            chunk_embeddings
        ):

            similarity = (
                self.cosine_similarity(
                    query_embedding,
                    embedding
                )
            )

            scores.append({
                "chunk": chunk,
                "score": similarity
            })

        scores.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return scores[:top_k]