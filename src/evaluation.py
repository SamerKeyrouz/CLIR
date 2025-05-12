from tqdm import tqdm
import pandas as pd
import numpy as np
import json

# Load QRELs
qrels_path = "../data/raw_data/2024-qrels.rus.with-gains.txt"
qrels_df = pd.read_csv(qrels_path, sep=" ", names=["qid", "Q0", "docid", "relevance"])

# Load processed document IDs
processed_docs_path = "../data/processed_data/russian_documents.jsonl"
your_doc_ids = set()
with open(processed_docs_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading processed docs"):
        doc = json.loads(line)
        your_doc_ids.add(doc["id"].strip())

# Get QREL document IDs
qrel_doc_ids = set(qrels_df["docid"].unique())
missing_docs = qrel_doc_ids - your_doc_ids

print(f"\n=== Document Verification ===")
print(f"Total QREL documents: {len(qrel_doc_ids)}")
print(f"Documents in your collection: {len(your_doc_ids)}")
print(f"Missing QREL documents: {len(missing_docs)}")
print("Sample missing docs:", list(missing_docs)[:3])

# Build qrels dictionary
qrels_dict = {}
for _, row in qrels_df.iterrows():
    qid = str(row["qid"]).strip()
    docid = row["docid"].strip()
    rel = int(row["relevance"])
    if qid not in qrels_dict:
        qrels_dict[qid] = {}
    qrels_dict[qid][docid] = rel

# Data Alignment Checks
print("\n=== Data Alignment Checks ===")
print("Sample QREL QIDs:", sorted(qrels_dict.keys())[:5])
print("Total unique QIDs in QRELs:", len(qrels_dict))
relevant_counts = [len(v) for v in qrels_dict.values()]
print("\nRelevant docs per query:")
print(pd.Series(relevant_counts).describe())
print(f"Queries with 0 relevant docs: {sum(1 for c in relevant_counts if c == 0)}")

# Helper to check overlap
def check_qid_overlap(run_df, run_name):
    run_qids = set(str(qid).strip() for qid in run_df['qid'].unique())
    qrel_qids = set(qrels_dict.keys())
    print(f"\n-- {run_name} --")
    print(f"Unique QIDs in run: {len(run_qids)}")
    print(f"QIDs matching QRELs: {len(run_qids & qrel_qids)}")
    print("Sample mismatched QIDs:", list(run_qids - qrel_qids)[:3])

# Load run files
bm25_df = pd.read_csv("../results/bm25_results.trec", sep=" ", names=["qid", "Q0", "docid", "rank", "score", "method"])
bm25_df['qid'] = bm25_df['qid'].astype(str).str.strip()

faiss_df = pd.read_csv("../results/retrieval_results.trec", sep=" ", names=["qid", "Q0", "docid", "rank", "score", "method"])
faiss_df['qid'] = faiss_df['qid'].astype(str).str.strip()

hybrid_df = pd.read_csv("../results/hybrid_results.trec", sep=" ", names=["qid", "Q0", "docid", "rank", "score", "method"])
hybrid_df['qid'] = hybrid_df['qid'].astype(str).str.strip()

check_qid_overlap(bm25_df, "BM25")
check_qid_overlap(faiss_df, "FAISS")
check_qid_overlap(hybrid_df, "Hybrid")

# Manual check
sample_qid = next(iter(qrels_dict))
print(f"\n-- Manual Check for QID {sample_qid} --")
print(f"Relevant docs in QRELs: {list(qrels_dict[sample_qid].keys())[:3]}")
print(f"BM25 retrieved docs: {bm25_df[bm25_df['qid'] == sample_qid]['docid'].tolist()[:5]}")
print(f"CLIR retrieved docs: {faiss_df[faiss_df['qid'] == sample_qid]['docid'].tolist()[:5]}")

# Metrics
def precision_at_k(retrieved, relevant, k=5):
    return sum(doc in relevant for doc in retrieved[:k]) / k

def average_precision(retrieved, relevant):
    hits, sum_prec = 0, 0.0
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(relevant) if relevant else 0.0

def recall_at_k(retrieved, relevant, k=1000):
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant) if relevant else 0.0

def ndcg_at_k(retrieved, relevant_rels, k):
    dcg = sum((2 ** relevant_rels.get(doc, 0) - 1) / np.log2(i + 2)
              for i, doc in enumerate(retrieved[:k]))
    ideal = sorted(relevant_rels.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

# Evaluation loop
def evaluate(run_df, method_name):
    run_df["qid"] = run_df["qid"].astype(str)
    all_prec, all_map, all_ndcg5, all_ndcg100, all_recall = [], [], [], [], []

    for qid in run_df["qid"].unique():
        if qid not in qrels_dict:
            continue
        retrieved_docs = run_df[run_df["qid"] == qid].sort_values("rank")["docid"].tolist()
        relevant_docs = list(qrels_dict[qid].keys())
        relevant_rels = qrels_dict[qid]

        all_prec.append(precision_at_k(retrieved_docs, relevant_docs, k=5))
        all_map.append(average_precision(retrieved_docs, relevant_docs))
        all_ndcg5.append(ndcg_at_k(retrieved_docs, relevant_rels, k=5))
        all_ndcg100.append(ndcg_at_k(retrieved_docs, relevant_rels, k=100))
        all_recall.append(recall_at_k(retrieved_docs, relevant_docs, k=1000))

    if not all_prec:
        print(f"No QID overlap with QRELs in {method_name} run!")
        return

    print(f"\nEvaluation for {method_name}:")
    print(f"  Precision@5:  {np.mean(all_prec):.4f}")
    print(f"  Recall@1000:  {np.mean(all_recall):.4f}")
    print(f"  MAP:          {np.mean(all_map):.4f}")
    print(f"  NDCG@5:       {np.mean(all_ndcg5):.4f}")
    print(f"  NDCG@100:     {np.mean(all_ndcg100):.4f}")

# Run evaluation
evaluate(bm25_df, "BM25_Elastic")
evaluate(faiss_df, "FAISS")
evaluate(hybrid_df, "Hybrid")
