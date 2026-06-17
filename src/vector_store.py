import chromadb

class VectorStore:
    def __init__(self, persist_directory="chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name="documents")

    def add_chunks(self, chunks, embeddings):
        ids =[]
        documents=[]
        metadatas=[]
        embedding_list = []

        for chunk, embedding in zip(chunks, embeddings):
            ids.append(str(chunk['chunk_id']))
            documents.append(chunk['chunk'])
            metadatas.append({
                "chunk_id": chunk['chunk_id'],
                "document_name": chunk['document_name'],
                "page_number": chunk['page_number']
            })
            embedding_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embedding_list
        )

    def search(self,query_embedding,top_k=5):

        results = (
            self.collection.query(
                query_embeddings=[
                    query_embedding.tolist()
                ],
                n_results=top_k
            )
        )

        output = []
        for idx, doc in enumerate(results["documents"][0]):
            output.append({
                "chunk_id": results["metadatas"][0][idx]["chunk_id"],
                "document_name": results["metadatas"][0][idx]["document_name"],
                "page_number": results["metadatas"][0][idx]["page_number"],
                "chunk": doc,
                "distance": results["distances"][0][idx]
            })

        return output
    
    def reset(self):

        try:

            self.client.delete_collection(
                "documents"
            )

        except Exception:
            pass

        self.collection = (
            self.client.get_or_create_collection(
                name="documents"
            )
        )