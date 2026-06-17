class ErrorAnalyzer:

    def save_failures(self,benchmark_df):
        failures = benchmark_df[benchmark_df["hit_rate"] == 0]
        failures.to_csv("results/failure_analysis.csv", index=False)
        return failures