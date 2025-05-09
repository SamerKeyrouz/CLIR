import faiss
import numpy as np
import os


def main():
    # Path configuration
    DOC_EMBEDDINGS_PATH = "../data/embeddings/russian_docs.npy"
    QUERY_EMBEDDINGS_PATH = "../data/embeddings/english_queries.npy"
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

    # 2. Normalize embeddings for cosine similarity
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
        for q_id, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            for rank, (score, doc_idx) in enumerate(zip(query_scores, query_indices)):
                f.write(f"{q_id + 1} Q0 {doc_idx} {rank + 1} {score:.6f} CLIR_Project\n")

    print(f"\nSaved results to {RESULTS_PATH}")
    print("Retrieval complete!")


if __name__ == "__main__":
    main()