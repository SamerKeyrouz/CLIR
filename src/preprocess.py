import spacy
import json

# Load the Russian spaCy model
nlp_ru = spacy.load('ru_core_news_sm')  # For Russian documents

# Load the English spaCy model (for queries, if needed)
nlp_en = spacy.load('en_core_web_sm')  # For English queries


# Function to load data in JSONL format and process a subset (line by line)
def load_jsonl(file_path, num_lines=None, chunk_size=1000):
    """
    Load data from a JSONL file in chunks to handle large files.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_lines and i >= num_lines:
                break  # Stop after reading 'num_lines' lines
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping invalid line: {line}")

            # Process in chunks to avoid memory overload
            if len(data) >= chunk_size:
                yield data  # Yield the current chunk
                data = []  # Reset the data list to free memory

    if data:  # Yield any remaining data after processing the file
        yield data


# Function to preprocess Russian documents
def preprocess_russian(doc):
    text = doc['text']
    # Process the document text using spaCy
    spacy_doc = nlp_ru(text)
    # Lemmatize and remove punctuation
    processed_text = " ".join([token.lemma_ for token in spacy_doc if not token.is_punct])
    return processed_text


# Function to preprocess topics (queries)
# Function to preprocess topics (queries)
# Function to preprocess topics (queries)
# Function to preprocess topics (queries)
# Function to preprocess topics (queries)
# Function to preprocess topics (queries)
# Function to preprocess topics (queries)
def preprocess_topics(topic):
    # Ensure you're accessing the 'topic_id' from the main topic dictionary
    topic_id = topic.get('topic_id', 'Unknown ID')

    # Check if 'topics' field exists and contains at least one item
    if 'topics' in topic and isinstance(topic['topics'], list) and len(topic['topics']) > 0:
        # Iterate over all topic variants in the 'topics' list
        for topic_variant in topic['topics']:
            lang = topic_variant.get('lang', '')

            # Process only English topics
            if lang != 'eng':
                print(f"Skipping non-English topic (lang={lang}): {topic_id}")
                continue  # Skip this variant and move to the next one

            text = topic_variant.get('topic_description', '')  # Use get() to avoid KeyError
            # Process the topic text using spaCy
            spacy_doc = nlp_en(text)
            # Lemmatize and remove punctuation
            processed_text = " ".join([token.lemma_ for token in spacy_doc if not token.is_punct])
            return processed_text  # Process only the first English topic variant

        # If no English topic variant found, skip
        print(f"Skipping topic with no English variant: {topic_id}")
        return ''
    else:
        # If 'topics' is missing or empty, skip this topic
        print(f"Skipping topic with no 'topics' field or empty list: {topic_id}")
        return ''


# Function to save preprocessed data into JSONL format
def save_preprocessed_data(file_path, data, mode='a'):
    """
    Save data to a JSONL file.
    - mode='a': Append mode (default)
    - mode='w': Overwrite mode
    """
    with open(file_path, mode, encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps({"text": item}) + '\n')


# Main processing function
def main():
    # Define how many documents to process (set to 1000 or more as needed)
    num_documents = 20000  # Adjust this value to control the subset size
    num_topics = None  # No limit on the number of topics (keep all)

    # Clear existing file (start fresh) (ADDED)
    open('../data/processed_data/russian_documents.jsonl', 'w').close()

    # Process documents in chunks
    for chunk in load_jsonl('../data/raw_data/rus/docs.jsonl', num_lines=num_documents, chunk_size=1000):
        # Preprocess each chunk of Russian documents
        processed_docs = [preprocess_russian(doc) for doc in chunk]
        # Save the preprocessed Russian documents
        save_preprocessed_data('../data/processed_data/russian_documents.jsonl', processed_docs,mode='a')
        print(f"Processed and saved {len(processed_docs)} documents.")

    # Load all topics (queries) by flattening chunks into a single list
    topics = []
    for chunk in load_jsonl('../data/raw_data/neuclir24.topics.0614.jsonl.txt', num_lines=num_topics):
        topics.extend(chunk)  # Flatten chunks into a single list
    print(f"Loaded {len(topics)} topics.")

    # Preprocess topics (queries)
    processed_topics = [preprocess_topics(topic) for topic in topics]

    # Save preprocessed topics
    save_preprocessed_data('../data/processed_data/processed_topics.jsonl', processed_topics)
    print(f"Saved {len(processed_topics)} processed topics.")


if __name__ == "__main__":
    main()
