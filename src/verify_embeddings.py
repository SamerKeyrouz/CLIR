import numpy as np

# Load embeddings
doc_emb = np.load("../data/embeddings/russian_docs.npy")
query_emb = np.load("../data/embeddings/english_queries.npy")

print("Document embeddings shape:", doc_emb.shape)  # Should be (1000, 768)
print("Query embeddings shape:", query_emb.shape)    # Should be (93, 768)