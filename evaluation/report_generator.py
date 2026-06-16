import os

class ReportGenerator:
    def generate(self, benchmark_df):
        os.makedirs("results",exist_ok=True)

        summary = f"""
            # Retrieval Benchmark Report

            Queries:
            {len(benchmark_df)}

            Average Hit Rate:
            {benchmark_df['hit_rate'].mean():.3f}

            Average Precision@3:
            {benchmark_df['precision@3'].mean():.3f}

            Average Recall:
            {benchmark_df['recall'].mean():.3f}

            Average MRR:
            {benchmark_df['mrr'].mean():.3f}
        """

        with open("results/benchmark_summary.md","w") as f:
            f.write(summary)