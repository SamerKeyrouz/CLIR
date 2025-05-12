from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import json

# Configuration
DOCUMENTS_PATH = "../data/processed_data/russian_documents.jsonl"
QUERIES_PATH = "../data/processed_data/processed_topics.jsonl"
RESULTS_PATH = "../results/bm25_results.trec"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"


# ✅ Translate English queries to Russian using MarianMT
def translate_queries(queries):
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)


    translated = []
    for i in tqdm(range(0, len(queries), 8), desc="Translating queries"):
        batch = queries[i:i+8]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**encoded, max_length=128)
        translated_batch = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
        translated.extend(translated_batch)

    # Print a few samples
    print("\nSample translated queries:")
    for i, q in enumerate(translated[:3]):
        print(f"Query {i+1}: {q}")
    return translated


def main():
    es = Elasticsearch(["http://localhost:9200"],request_timeout = 30)  # Increase timeout to 30 seconds)

    if es.indices.exists(index="russian_news"):
        print("Deleting old index...")
        es.indices.delete(index="russian_news", ignore_unavailable=True)
        es.indices.refresh()
        print("Index deleted.")
    else:
        print("No existing index to delete.")

    es.indices.create(
        index="russian_news",
        body={
            "settings": {
                "analysis": {
                    "analyzer": {
                        "russian_analyzer": {
                            "type": "russian",  # Built-in Russian analyzer
                            "stopwords": "_russian_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "russian_analyzer",
                        "search_analyzer": "russian_analyzer"
                    }
                }
            }
        }
    )

    # Index documents with their UUIDs
    documents = []
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=20000, desc="Indexing documents"):
            data = json.loads(line)
            documents.append({
                "_index": "russian_news",
                "_id": data["id"],
                "_source": {"text": data["text"]}
            })

    failed_ids = []

    for i in range(0, len(documents), 1000):
        chunk = documents[i:i + 1000]

        # Use raise_on_error=False to get error details
        success, errors = bulk(es, chunk, raise_on_error=False, stats_only=False)

        # Extract failed doc IDs (if any)
        for err in errors:
            doc_id = err.get('index', {}).get('_id')
            if doc_id:
                failed_ids.append(doc_id)

        print(f"Batch {i // 1000}: Success={success}, Failures={len(errors)}")

    # Final summary
    print(f"\n❌ Total failed documents: {len(failed_ids)}")
    if failed_ids:
        print("Sample failed IDs:", failed_ids[:5])

    es.indices.refresh(index="russian_news")
    print(f"\nTotal documents indexed: {es.count(index='russian_news')['count']}")

    # Load queries
    queries = []
    topic_ids = []
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries.append(obj["text"])
            topic_ids.append(obj["topic_id"])

    print(f"Loaded {len(queries)} queries from processed_topics.jsonl")
    # Translate queries
    translated_queries = translate_queries(queries)

    # Perform search
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for i, query in enumerate(tqdm(translated_queries, desc="Processing queries")):
            topic_id = topic_ids[i]
            res = es.search(
                index="russian_news",
                body={"query": {"match": {"text": query}}, "size": 1000}
            )
            for rank, hit in enumerate(res['hits']['hits']):
                f.write(f"{topic_id} Q0 {hit['_id']} {rank + 1} {hit['_score']:.6f} BM25_Elastic\n")


if __name__ == "__main__":
    main()
