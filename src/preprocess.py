import spacy
import json

# Load the Russian spaCy model
nlp_ru = spacy.load('ru_core_news_sm')

# Load the English spaCy model
nlp_en = spacy.load('en_core_web_sm')


# Load data in JSONL format (chunked for large files)
def load_jsonl(file_path, num_lines=None, chunk_size=1000):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_lines and i >= num_lines:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line}")

            if len(data) >= chunk_size:
                yield data
                data = []
    if data:
        yield data


# Preprocess a single Russian document
def preprocess_russian(doc):
    text = doc['text']
    doc_id = doc['id']  # Preserve the original UUID
    spacy_doc = nlp_ru(text)
    processed_text = " ".join([token.lemma_ for token in spacy_doc if not token.is_punct])
    return {"id": doc_id, "text": processed_text}


# Preprocess a single English topic
def preprocess_topics(topic):
    topic_id = topic.get('topic_id', 'Unknown ID')

    if 'topics' in topic and isinstance(topic['topics'], list) and len(topic['topics']) > 0:
        for topic_variant in topic['topics']:
            if topic_variant.get('lang', '') != 'eng':
                continue
            text = topic_variant.get('topic_description', '')
            spacy_doc = nlp_en(text)
            processed_text = " ".join([token.lemma_ for token in spacy_doc if not token.is_punct])
            return {"text": processed_text, "topic_id": topic_id}
        print(f"Skipping topic with no English variant: {topic_id}")
        return None
    else:
        print(f"Skipping topic with no 'topics' field or empty list: {topic_id}")
        return None


# Save data in JSONL format
def save_preprocessed_data(file_path, data, mode='a'):
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            if item is not None:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


# Main processing function
def main():
    num_documents = 20000
    num_topics = None

    # Clear previous output
    open('../data/processed_data/russian_documents.jsonl', 'w').close()
    open('../data/processed_data/processed_topics.jsonl', 'w').close()

    # Process Russian documents
    for chunk in load_jsonl('../data/raw_data/rus/docs.jsonl', num_lines=num_documents, chunk_size=1000):
        processed_docs = [preprocess_russian(doc) for doc in chunk]
        save_preprocessed_data('../data/processed_data/russian_documents.jsonl', processed_docs, mode='a')
        print(f"Processed and saved {len(processed_docs)} Russian documents.")

    # Process English topics
    topics = []
    for chunk in load_jsonl('../data/raw_data/neuclir24.topics.0614.jsonl.txt', num_lines=num_topics):
        topics.extend(chunk)

    print(f"Loaded {len(topics)} topics.")
    processed_topics = [preprocess_topics(topic) for topic in topics]
    save_preprocessed_data('../data/processed_data/processed_topics.jsonl', processed_topics)
    print(f"Saved {len([t for t in processed_topics if t is not None])} processed topics.")


if __name__ == "__main__":
    main()
