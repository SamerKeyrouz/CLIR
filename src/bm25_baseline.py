from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
from googletrans import Translator
from tqdm import tqdm

# Configuration
DOCUMENTS_PATH = "../data/processed_data/russian_documents.jsonl"
QUERIES_PATH = "../data/processed_data/processed_topics.jsonl"
RESULTS_PATH = "../results/bm25_results.trec"


def translate_queries(queries):
    """Translate English queries to Russian using googletrans"""
    translator = Translator()
    translated = []
    for q in tqdm(queries, desc="Translating queries"):
        try:
            translated.append(translator.translate(q, src='en', dest='ru').text)
        except:
            translated.append(q)  # Fallback to original if translation fails
    return translated


def main():
    # 1. Initialize Elasticsearch client
    es = Elasticsearch(["http://localhost:9200"])

    # 2. Create index with Russian analyzer
    if not es.indices.exists(index="russian_news"):
        es.indices.create(
            index="russian_news",
            body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "russian_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase", "russian_morphology", "english_stop"]
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

    # 3. Load and index documents
    print("Indexing documents...")
    documents = []
    with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=20000):  # Update total if different
            data = json.loads(line)
            documents.append({
                "_index": "russian_news",
                "_id": str(len(documents)),
                "_source": {"text": data["text"]}
            })

    # Bulk index in batches of 1000
    for i in range(0, len(documents), 1000):
        bulk(es, documents[i:i + 1000])

    # 4. Load and translate queries
    print("\nLoading and translating queries...")
    with open(QUERIES_PATH, "r") as f:
        queries = [json.loads(line)["text"] for line in f]

    translated_queries = translate_queries(queries)

    # 5. Search and save results
    print("\nSearching...")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for q_id, query in enumerate(tqdm(translated_queries, desc="Processing queries")):
            body = {
                "query": {
                    "match": {
                        "text": {
                            "query": query,
                            "operator": "and"  # Require all query terms
                        }
                    }
                },
                "size": 1000
            }
            res = es.search(index="russian_news", body=body)

            for rank, hit in enumerate(res['hits']['hits']):
                f.write(f"{q_id + 1} Q0 {hit['_id']} {rank + 1} {hit['_score']:.6f} BM25_Elastic\n")

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()