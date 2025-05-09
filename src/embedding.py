import numpy as np
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

# Configuration
DOCUMENTS_PATH = "../data/processed_data/russian_documents.jsonl"
TOPICS_PATH = "../data/processed_data/processed_topics.jsonl"
MODEL_NAME = "sentence-transformers/LaBSE"  # Best for CLIR
BATCH_SIZE = 256  # Adjust based on your GPU memory


def load_texts(file_path):
    """Load preprocessed texts from JSONL"""
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {file_path}"):
            texts.append(json.loads(line)["text"])
    return texts


def generate_embeddings(model, texts):
    """Generate embeddings in batches"""
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )


def main():
    # 1. Load model
    model = SentenceTransformer(MODEL_NAME)

    # 2. Load preprocessed data
    print("Loading documents...")
    russian_docs = load_texts(DOCUMENTS_PATH)

    print("\nLoading queries...")
    english_queries = load_texts(TOPICS_PATH)

    # 3. Generate embeddings
    print("\nGenerating document embeddings...")
    doc_embeddings = generate_embeddings(model, russian_docs)

    print("\nGenerating query embeddings...")
    query_embeddings = generate_embeddings(model, english_queries)

    # 4. Save embeddings
    np.save("../data/embeddings/russian_docs.npy", doc_embeddings)
    np.save("../data/embeddings/english_queries.npy", query_embeddings)
    print("\nSaved embeddings to ../data/embeddings/")


if __name__ == "__main__":
    main()