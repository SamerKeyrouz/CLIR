import numpy as np
import json


# Function to load relevance judgments (qrels file)
def load_qrels(qrels_file_path):
    qrels = {}
    with open(qrels_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            topic_id, doc_id, relevance = parts[0], parts[1], int(parts[2])
            if topic_id not in qrels:
                qrels[topic_id] = {}
            qrels[topic_id][doc_id] = relevance
    return qrels


# Function to compute Precision@k
def precision_at_k(retrieved, relevant, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = set(retrieved_at_k) & set(relevant)
    return len(relevant_at_k) / k


# Function to compute Average Precision (AP) for a single query
def average_precision(retrieved, relevant):
    ap = 0.0
    num_relevant = 0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            num_relevant += 1
            ap += num_relevant / (i + 1)
    return ap / len(relevant) if relevant else 0.0


# Function to compute Mean Average Precision (MAP)
def mean_average_precision(retrieved_results, qrels):
    ap_sum = 0.0
    num_queries = len(qrels)

    for topic_id, relevant_docs in qrels.items():
        retrieved = retrieved_results.get(topic_id, [])
        ap_sum += average_precision(retrieved, list(relevant_docs.keys()))

    return ap_sum / num_queries if num_queries else 0.0


# Function to compute NDCG
def ndcg_at_k(retrieved, relevant, k):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += (2 ** relevant[doc_id] - 1) / np.log2(i + 2)

    ideal_dcg = 0.0
    ideal_relevant_docs = sorted(relevant.values(), reverse=True)
    for i, rel in enumerate(ideal_relevant_docs[:k]):
        ideal_dcg += (2 ** rel - 1) / np.log2(i + 2)

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# Function to compute NDCG for all queries
def mean_ndcg(retrieved_results, qrels, k=10):
    ndcg_sum = 0.0
    num_queries = len(qrels)

    for topic_id, relevant_docs in qrels.items():
        retrieved = retrieved_results.get(topic_id, [])
        ndcg_sum += ndcg_at_k(retrieved, relevant_docs, k)

    return ndcg_sum / num_queries if num_queries else 0.0


# Main evaluation function
def main():
    # Load the relevance judgments
    qrels = load_qrels('data/raw_data/2024-qrels.rus.with-gains.txt')

    # Sample retrieved results (format: {topic_id: [doc_id1, doc_id2, ..., doc_idk]})
    retrieved_results = {
        '1': ['doc1', 'doc2', 'doc3'],  # Example, use actual retrieval results
        '2': ['doc5', 'doc6', 'doc7'],
        # Add all queries here
    }

    # Compute MAP
    map_score = mean_average_precision(retrieved_results, qrels)
    print(f"Mean Average Precision (MAP): {map_score:.4f}")

    # Compute Precision@k (example: k=10)
    k = 10
    precision_scores = {}
    for topic_id, relevant_docs in qrels.items():
        precision_scores[topic_id] = precision_at_k(retrieved_results.get(topic_id, []), list(relevant_docs.keys()), k)

    print(f"Precision@{k}: {np.mean(list(precision_scores.values())):.4f}")

    # Compute NDCG
    ndcg_score = mean_ndcg(retrieved_results, qrels, k)
    print(f"Mean NDCG@{k}: {ndcg_score:.4f}")


if __name__ == "__main__":
    main()
