from src.embedder import Embedder
embedder = Embedder()
sentences  = [
    "What is backpropagation?",
    "How do neural networks learn?",
    "I like pizza."
]

vectors = embedder.embed_chunks(
    [{"chunk": sentence} for sentence in sentences]
)

print(vectors.shape)
print(vectors[0][:5])
print(vectors[1][:5])
print(vectors[2][:5])
from sklearn.metrics.pairwise import cosine_similarity

sim12 = cosine_similarity([vectors[0]], [vectors[1]])  # similar sentences
sim13 = cosine_similarity([vectors[0]], [vectors[2]])  # different sentence

print(sim12, sim13)