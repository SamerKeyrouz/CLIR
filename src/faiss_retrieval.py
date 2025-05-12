import faiss
import numpy as np
import os


def main():
    # Path configuration
    DOC_EMBEDDINGS_PATH = "../data/embeddings/russian_docs.npy"
    QUERY_EMBEDDINGS_PATH = "../data/embeddings/english_queries.npy"
    DOC_IDS_PATH = "../data/embeddings/doc_ids.txt"
    QUERY_IDS_PATH = "../data/embeddings/query_ids.txt"
    FAISS_INDEX_PATH = "../data/faiss_index/russian_docs.faiss"
    RESULTS_PATH = "../results/retrieval_results.trec"

    # Create necessary directories
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    # 1. Load embeddings
    print("Loading document embeddings...")
    doc_embeddings = np.load(DOC_EMBEDDINGS_PATH).astype(np.float32)

    print("Loading query embeddings...")
    query_embeddings = np.load(QUERY_EMBEDDINGS_PATH).astype(np.float32)

    # ✅ Load real document IDs
    with open(DOC_IDS_PATH, "r", encoding="utf-8") as f:
        doc_ids = [line.strip() for line in f]

    # ✅ Load real topic IDs
    with open(QUERY_IDS_PATH, "r", encoding="utf-8") as f:
        query_ids = [line.strip() for line in f]

    # Sanity checks
    assert len(doc_ids) == doc_embeddings.shape[0], "Mismatch between doc IDs and embeddings!"
    assert len(query_ids) == query_embeddings.shape[0], "Mismatch between query IDs and embeddings!"

    # 2. Normalize embeddings
    print("\nNormalizing embeddings...")
    faiss.normalize_L2(doc_embeddings)
    faiss.normalize_L2(query_embeddings)

    # 3. Build and save FAISS index
    print("\nBuilding FAISS index...")
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine Similarity
    index.add(doc_embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Index built with {index.ntotal} documents")

    # 4. Perform retrieval
    print("\nPerforming search...")
    scores, indices = index.search(query_embeddings, 1000)

    # 5. Save results in TREC format
    print("\nSaving results...")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for i, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            topic_id = query_ids[i]  # ✅ use actual topic ID
            for rank, (score, doc_idx) in enumerate(zip(query_scores, query_indices)):
                real_doc_id = doc_ids[doc_idx]
                f.write(f"{topic_id} Q0 {real_doc_id} {rank + 1} {score:.6f} CLIR_Project\n")

    print(f"\nSaved results to {RESULTS_PATH}")
    print("Retrieval complete!")


if __name__ == "__main__":
    main()
