import numpy as np
from chunker import chunk_wiki_files
import re
import math
from topic_expansion_data import topic_expansion
import argparse
import sys
import os
import json
import pickle

import ollama

# step 1 : vocabulary creation

CACHE_DIR = ".cache"
CACHE_META = os.path.join(CACHE_DIR, "vectorize_meta.json")
CACHE_DATA = os.path.join(CACHE_DIR, "vectorize_cache.pkl")

chunks = []
vocab = {}
document_vectors = []
idf_vectors = []
tfidf_vectors = []

def sanitize(str: str):
    str = str.lower()
    str = str.replace("\n", " ")
    str = re.sub(r'[^\w\s-]', '', str)
    str = re.sub(r'\s+', ' ', str).strip()
    return str

def build_vocab(chunks):
    # creates a pair of {word, id} such as we get unique words with their id 
    # duplicates wont be storred
    word_to_id = {}
    next_id = 0

    for chunk in chunks:
        tokens = chunk['text'].split(' ')
        for token in tokens:
            sanitized = sanitize(token)
            if sanitized not in word_to_id:
                word_to_id[sanitized] = next_id
                next_id += 1

    return word_to_id

# step 2 : document vector

def document_vector(chunks, vocab):
    #create a bag of words from build_vocab
    #map id's to translate doc into numeric vector
    all_vectors = []
    
    for chunk in chunks:
        vector = np.zeros(len(vocab))
        words = chunk['text'].split(' ')
        for word in words:
            if word in vocab:
                word_id = vocab[word]
                vector[word_id] += 1
        all_vectors.append(vector)
    return all_vectors


# step 3: Cosine Similarity Search
# we measure the similarity between two chunks by calculating their angle
# similarity = (A . B) / (|A| x |B|)

def query_to_vector(query):
    # take a query and return it into a vector
    tokens = sanitize(query)
    tokens = tokens.split(' ' or ',' or '.')
    vector = np.zeros(len(vocab))

    for token in tokens:
        if token in vocab:
            token_index_in_vocab = vocab[token]
            vector[token_index_in_vocab] = 1
        
    return vector




# step 4 : TF-IDF Vactorizing 

# def idf_word(vocab=vocab, chunks = chunks): 

#     frequency = 0
#     idf = 0
#     vocabulary_frequency = {'idf': idf, 'frequency':frequency}
#     for chunk in chunks:
#         tokinized_chunk = chunk['text'].split(' ')
#         for chunk in tokinized_chunk:
#             if vocab[chunk] in chunk:
#                 vocabulary_frequency[vocab['id']][frequency] += 1
#                 vocabulary_frequency[vocab['id']][frequency] = math.log(len(chunks)/ (1+vocabulary_frequency[vocab['id']][frequency]))
#     print(vocabulary_frequency)
    
def doc_freq(vocab, chunks):
    
    doc_freq = np.zeros(len(vocab))
    for chunk in chunks: 
        
        words = chunk['text'].split(' ')
        unique_words = set(words)
        for word in unique_words: 
            if word in vocab:
                word_id = vocab[word]
                doc_freq[word_id] += 1
    return doc_freq

def idf(doc_freq_vec, total_chunks):
    idf_vec = np.zeros(len(doc_freq_vec))

    for word_id, freq in enumerate(doc_freq_vec):
        if freq>0:
            idf_vec[word_id] = np.log(total_chunks/freq)
        else:
            idf_vec[word_id] = 0

    return idf_vec

def get_wiki_signature():
    wiki_dir = "wiki"
    if not os.path.isdir(wiki_dir):
        return {"count": 0, "latest_mtime": 0, "total_size": 0}
    files = [
        os.path.join(wiki_dir, f)
        for f in os.listdir(wiki_dir)
        if f.endswith(".txt")
    ]
    if not files:
        return {"count": 0, "latest_mtime": 0, "total_size": 0}
    stats = [os.stat(f) for f in files]
    return {
        "count": len(files),
        "latest_mtime": max(s.st_mtime for s in stats),
        "total_size": sum(s.st_size for s in stats),
    }

def load_cache(signature):
    if not (os.path.exists(CACHE_META) and os.path.exists(CACHE_DATA)):
        return None
    try:
        with open(CACHE_META, "r", encoding="utf-8") as meta_file:
            cached_sig = json.load(meta_file)
        if cached_sig != signature:
            return None
        with open(CACHE_DATA, "rb") as data_file:
            return pickle.load(data_file)
    except (OSError, json.JSONDecodeError, pickle.UnpicklingError):
        return None

def save_cache(signature, payload):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_META, "w", encoding="utf-8") as meta_file:
        json.dump(signature, meta_file)
    with open(CACHE_DATA, "wb") as data_file:
        pickle.dump(payload, data_file)

def initialize_index():
    global chunks, vocab, document_vectors, idf_vectors, tfidf_vectors
    signature = get_wiki_signature()
    cache = load_cache(signature)
    if cache:
        chunks = cache["chunks"]
        vocab = cache["vocab"]
        document_vectors = cache["document_vectors"]
        idf_vectors = cache["idf_vectors"]
        tfidf_vectors = cache["tfidf_vectors"]
        return

    chunks = chunk_wiki_files()
    vocab = build_vocab(chunks)
    document_vectors = document_vector(chunks, vocab)
    doc_freq_vec = doc_freq(vocab, chunks)
    idf_vectors = idf(doc_freq_vec, len(chunks))
    tfidf_vectors = [tf_vector * idf_vectors for tf_vector in document_vectors]

    save_cache(
        signature,
        {
            "chunks": chunks,
            "vocab": vocab,
            "document_vectors": document_vectors,
            "idf_vectors": idf_vectors,
            "tfidf_vectors": tfidf_vectors,
        },
    )

def query_to_idf_vector(query):
    # take a query and return it into a vector
    tokens = sanitize(query)
    tokens = tokens.split(' ' or ',' or '.')
    vector = np.zeros(len(vocab))

    for token in tokens:
        if token in vocab:
            token_index_in_vocab = vocab[token]
            vector[token_index_in_vocab] = idf_vectors[token_index_in_vocab]

            if token in topic_expansion:
                for expaneded_word in topic_expansion[token]:
                    clean_word = sanitize(expaneded_word)
                    if clean_word in vocab:
                        exp_id = vocab[clean_word]
                        # give it half weight since its expanded
                        vector[exp_id] += idf_vectors[exp_id] * 0.5 
        
    return vector

def search(query_vector, chunks_vector=tfidf_vectors, n = 3):
    # array index represents the chunk index
    # array value represents the cosine similarity


    compare_array = []
    for chunk_vector in chunks_vector: 
        dot_product = np.dot(chunk_vector, query_vector)
        norm1 = np.linalg.norm(chunk_vector)
        norm2 = np.linalg.norm(query_vector)

        if norm1 != 0 and norm2 != 0:
            similarity = dot_product / (norm1 * norm2)
            compare_array.append(similarity)
        
        else:
            compare_array.append(0)
    sorted_indices = np.argsort(compare_array)[::-1]
    top_indices = sorted_indices[:n]

    top_n_matches = [(chunks[i], compare_array[i]) for i in top_indices]
    return top_n_matches

def generate_with_llm(query, top_chunks):
    context = "\n\n".join([chunk['text'] for chunk, _ in top_chunks])
    
    response = ollama.chat(
        model='llama3.2:3b',  
        messages=[
            {'role': 'system', 'content': 'Answer concisely based on the Wikipedia excerpts provided.'},
            {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        options={'num_predict': 256}  
    )
    return response['message']['content']

if __name__ == "__main__": 
    if len(sys.argv) <= 1:
        print("usage: python3 vectorize.py [-llm] your query")
        sys.exit(0)

    llm_mode = False
    args = sys.argv[1:]
    if args and args[0] == "-llm":
        llm_mode = True
        args = args[1:]

    if not args:
        print("usage: python3 vectorize.py [-llm] your query")
        sys.exit(0)

    query = " ".join(args)

    initialize_index()

    query_vector = query_to_idf_vector(query)
    result = search(query_vector, chunks_vector=tfidf_vectors, n=5)

    if llm_mode:
        answer = generate_with_llm(query, result)
        print(answer)
    else:
        for i, (chunk, score) in enumerate(result,1):
            print(f"{i}. [{score:.4f}] {chunk['source']}" )
            print(f"    {chunk['text']}")