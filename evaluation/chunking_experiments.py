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

CONFIGS = [

    (200, 50),

    (500, 100),

    (1000, 200)
]

os.makedirs(
    "results",
    exist_ok=True
)

golden_queries = pd.read_csv(
    "data/evaluation/golden_queries.csv"
)

results = []

for chunk_size, overlap in CONFIGS:

    print(
        f"\nTesting {chunk_size}/{overlap}"
    )

    pipeline = IndexingPipeline(

        pdf_path=PDF_PATH,

        chunk_size=chunk_size,

        overlap=overlap
    )

    (
        chunks,
        embeddings,
        embedder,
        vector_store

    ) = pipeline.build()

    retriever = Retriever(
        embedder,
        vector_store
    )

    benchmark = RetrievalBenchmark()

    benchmark_df = benchmark.run(
        golden_queries,
        retriever
    )

    results.append({

        "chunk_size":
            chunk_size,

        "overlap":
            overlap,

        "hit_rate":
            benchmark_df[
                "hit_rate"
            ].mean(),

        "precision@3":
            benchmark_df[
                "precision@3"
            ].mean(),

        "recall":
            benchmark_df[
                "recall"
            ].mean(),

        "mrr":
            benchmark_df[
                "mrr"
            ].mean()
    })

results_df = pd.DataFrame(
    results
)

results_df.to_csv(

    "results/chunking_results.csv",

    index=False
)

print(results_df)