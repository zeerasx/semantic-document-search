import os
import pandas as pd

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever

from evaluation.benchmark import RetrievalBenchmark
from evaluation.error_analysis import ErrorAnalyzer
from evaluation.report_generator import ReportGenerator

os.makedirs(
    "results",
    exist_ok=True
)

golden_queries = pd.read_csv("data/evaluation/golden_queries.csv")

embedder = Embedder()

vector_store = VectorStore()

retriever = Retriever(embedder,vector_store)

benchmark = RetrievalBenchmark()

benchmark_df = benchmark.run(golden_queries,retriever)

benchmark_df.to_csv(
    "results/benchmark_results.csv",
    index=False
)

ErrorAnalyzer().save_failures(benchmark_df)

ReportGenerator().generate(benchmark_df)

print(benchmark_df.mean(numeric_only=True))