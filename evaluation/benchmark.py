import pandas as pd

from evaluation.metrics import (
    hit_rate,
    precision_at_k,
    recall,
    reciprocal_rank
)


class RetrievalBenchmark:

    def run(
        self,
        golden_queries_df,
        retriever
    ):

        results = []

        for _, row in (golden_queries_df.iterrows()):
            query = row["query"]
            expected = row[
                "expected_keyword"
            ]

            search_results = (
                retriever.search(
                    query=query,
                    top_k=5
                )
            )

            retrieved_texts = [
                r["chunk"]
                for r in search_results
            ]

            results.append({
                "query":query,
                "expected":expected,

                "hit_rate":
                    hit_rate(retrieved_texts,expected),

                "precision@3":
                    precision_at_k(retrieved_texts,expected,3),

                "recall":
                    recall(retrieved_texts,expected),

                "mrr":
                    reciprocal_rank(retrieved_texts,expected)
            })

        return pd.DataFrame(
            results
        )