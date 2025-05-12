import json
from tqdm import tqdm
from preprocess import preprocess_russian


# Configuration
RAW_DOCS_PATH = "../data/raw_data/rus/docs.jsonl"
PROCESSED_DOCS_PATH = "../data/processed_data/russian_documents.jsonl"
QRELS_PATH = "../data/raw_data/2024-qrels.rus.with-gains.txt"


def main():
    # 1. Load existing processed IDs
    processed_ids = set()
    try:
        with open(PROCESSED_DOCS_PATH, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="🔄 Loading existing processed IDs"):
                processed_ids.add(json.loads(line)["id"])
        print(f"✅ Loaded {len(processed_ids)} existing document IDs")
    except FileNotFoundError:
        print("⚠️ No existing processed file found - starting fresh")

    # 2. Identify missing relevant docs from QRELs
    missing_docs = set()
    with open(QRELS_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="🔍 Analyzing QRELs"):
            parts = line.strip().split()
            if len(parts) >= 3:
                doc_id = parts[2]
                if doc_id not in processed_ids:
                    missing_docs.add(doc_id)

    print(f"\n📊 Found {len(missing_docs)} missing document IDs in QRELs")

    # 3. Find missing docs in raw data
    found_docs = []
    if missing_docs:
        with open(RAW_DOCS_PATH, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="🔎 Scanning raw documents"):
                doc = json.loads(line)
                if doc["id"] in missing_docs:
                    found_docs.append(doc)
                    missing_docs.remove(doc["id"])
                    if not missing_docs:
                        break

    # 4. Append to processed file with duplicate protection
    new_count = 0
    if found_docs:
        print(f"\n➕ Preparing to add {len(found_docs)} new documents")
        with open(PROCESSED_DOCS_PATH, "a", encoding="utf-8") as f:  # Append mode
            for doc in tqdm(found_docs, desc="📥 Appending documents"):
                # Double-check for duplicates
                if doc["id"] in processed_ids:
                    print(f"⚠️ Skipping duplicate: {doc['id']}")
                    continue

                # Preprocess and write
                processed = preprocess_russian(doc)  # Your existing function
                f.write(json.dumps(processed, ensure_ascii=False) + "\n")
                processed_ids.add(doc["id"])
                new_count += 1

        print(f"\n🎉 Successfully added {new_count} new documents")
        print(f"📂 Total documents now: {len(processed_ids)}")
    else:
        print("\n🤷 No new documents to add")

    # 5. Report missing docs not found
    if missing_docs:
        print(f"\n❌ Couldn't find {len(missing_docs)} documents in raw data:")
        for doc_id in list(missing_docs)[:5]:
            print(f" - {doc_id}")
        print(f"(Showing first 5 of {len(missing_docs)})")


if __name__ == "__main__":
    main()