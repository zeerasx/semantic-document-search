import os
import pandas as pd

from src.pipeline import IndexingPipeline
from src.retriever import Retriever

from evaluation.benchmark import (
    RetrievalBenchmark
)

PDF_PATH = (
    "data/pdfs/DeepLearningforDummies.pdf"
)

MODELS = [
    "all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2"
]

os.makedirs(
    "results",
    exist_ok=True
)

golden_queries = pd.read_csv(
    "data/evaluation/golden_queries.csv"
)

results = []

for model_name in MODELS:
    print(f"\nTesting {model_name}")

    pipeline = IndexingPipeline(
        pdf_path=PDF_PATH,
        chunk_size=500,
        overlap=100,
        model_name=model_name
    )

    (
        chunks,
        embeddings,
        embedder,
        vector_store

    ) = pipeline.build()

    retriever = Retriever(embedder,vector_store)

    benchmark = RetrievalBenchmark()

    benchmark_df = benchmark.run(golden_queries,retriever)

    results.append({
        "model":model_name,

        "hit_rate":benchmark_df["hit_rate"].mean(),

        "precision@3":benchmark_df["precision@3"].mean(),

        "recall":benchmark_df["recall"].mean(),

        "mrr":benchmark_df["mrr"].mean()
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/embedding_results.csv",
    index=False
)

print(results_df)