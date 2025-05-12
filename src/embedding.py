import numpy as np
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

# Configuration
DOCUMENTS_PATH = "../data/processed_data/russian_documents.jsonl"
TOPICS_PATH = "../data/processed_data/processed_topics.jsonl"
MODEL_NAME = "sentence-transformers/LaBSE"
BATCH_SIZE = 256


# ✅ Load both text and ID from Russian documents
def load_texts_and_ids(file_path):
    texts = []
    ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {file_path}"):
            obj = json.loads(line)
            texts.append(obj["text"])
            ids.append(obj["id"])
    return texts, ids


# ✅ Load English queries and their topic_id
def load_queries_with_ids(file_path):
    texts = []
    topic_ids = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {file_path}"):
            obj = json.loads(line)
            texts.append(obj["text"])
            topic_ids.append(obj.get("topic_id"))
    return texts, topic_ids


def generate_embeddings(model, texts):
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )


def main():
    # 1. Load model
    model = SentenceTransformer(MODEL_NAME)

    # 2. Load Russian documents
    print("Loading documents...")
    russian_texts, doc_ids = load_texts_and_ids(DOCUMENTS_PATH)

    # 3. Load English queries
    print("\nLoading queries...")
    english_queries, query_ids = load_queries_with_ids(TOPICS_PATH)

    # 4. Generate embeddings
    print("\nGenerating document embeddings...")
    doc_embeddings = generate_embeddings(model, russian_texts)

    print("\nGenerating query embeddings...")
    query_embeddings = generate_embeddings(model, english_queries)

    # 5. Save embeddings
    np.save("../data/embeddings/russian_docs.npy", doc_embeddings)
    np.save("../data/embeddings/english_queries.npy", query_embeddings)

    # ✅ Save document UUIDs
    with open("../data/embeddings/doc_ids.txt", "w", encoding="utf-8") as f:
        for doc_id in doc_ids:
            f.write(doc_id + "\n")

    # ✅ Save topic IDs (aligned with query embeddings)
    with open("../data/embeddings/query_ids.txt", "w", encoding="utf-8") as f:
        for qid in query_ids:
            f.write(str(qid) + "\n")

    # Optional debug
    print(f"\nSaved {len(doc_ids)} document IDs and {len(query_ids)} query IDs.")
    print("Doc embedding shape:", doc_embeddings.shape)
    print("Query embedding shape:", query_embeddings.shape)


if __name__ == "__main__":
    main()
