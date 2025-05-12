import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
BM25_RESULTS = "../results/bm25_results.trec"
CLIR_RESULTS = "../results/retrieval_results.trec"
HYBRID_RESULTS = "../results/hybrid_results.trec"
ALPHA = 0.7  # Weight for BM25 (1-ALPHA for CLIR)


def load_results(file_path):
    """Load TREC results into DataFrame"""
    return pd.read_csv(
        file_path,
        sep=" ",
        names=["qid", "Q0", "docid", "rank", "score", "method"],
        dtype={"qid": str, "docid": str}
    )


def main():
    print("🚀 Loading BM25 and CLIR results...")

    # Load both result sets
    bm25_df = load_results(BM25_RESULTS)
    clir_df = load_results(CLIR_RESULTS)

    print("\n🔗 Combining results using alpha={}...".format(ALPHA))

    # Normalize scores per query
    for df in [bm25_df, clir_df]:
        df["norm_score"] = df.groupby("qid")["score"].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

    # Merge results on query and document
    merged = pd.merge(
        bm25_df,
        clir_df,
        on=["qid", "docid"],
        suffixes=("_bm25", "_clir"),
        how="outer"
    ).fillna(0)

    # Calculate hybrid score
    merged["hybrid_score"] = (
            ALPHA * merged["norm_score_bm25"] +
            (1 - ALPHA) * merged["norm_score_clir"]
    )

    # Sort and rank
    merged = merged.sort_values(
        by=["qid", "hybrid_score"],
        ascending=[True, False]
    )

    # Generate final rankings
    print("\n💾 Saving hybrid results...")
    with open(HYBRID_RESULTS, "w") as f:
        for (qid, group) in tqdm(merged.groupby("qid"), desc="Processing queries"):
            for rank, (_, row) in enumerate(group.iterrows(), 1):
                f.write(
                    f"{qid} Q0 {row['docid']} {rank} {row['hybrid_score']:.6f} Hybrid\n"
                )

    print("\n✅ Hybrid results saved to", HYBRID_RESULTS)


if __name__ == "__main__":
    main()