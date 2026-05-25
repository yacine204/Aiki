# Aiki

Wikipedia **RAG system** (Retrieval-Augmented Generation) with a custom TF-IDF retriever built from scratch.

https://github.com/user-attachments/assets/992d43ed-8369-4032-a910-d10c80d528a5

## What it does

- Downloads Wikipedia articles into wiki/
- Splits documents into 500-character chunks
- Builds TF-IDF vectors from scratch (vocabulary size ~21k)
- Searches using cosine similarity (custom implementation)
- Supports query expansion via Wikipedia redirects/links
- **Optional**: Generates answers using local LLM (Ollama + Llama 3.2)

## Components

| Component | Built by |
|-----------|----------|
| Retriever (TF-IDF, vectors, similarity) | custom |
| Query expansion | custom |
| LLM generation | Ollama + Llama 3.2 |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install requests numpy langchain-text-splitters
pip install ollama
```

## Usage

Download articles:

```bash
python3 loader.py
```

Fast search (returns relevant chunks, milliseconds):

```bash
python3 vectorize.py "what is an android"
```

RAG mode (generates answer using local LLM, 5-15 seconds):

```bash
python3 vectorize.py -llm "what is an android"
```

Generate topic expansion data:

```bash
python3 topic_expansion.py --wiki-dir wiki
```

## Output examples

Fast search (your retriever):

```text
$ python3 vectorize.py "android"

1. [0.401] List of fictional robots and androids
   Human Torch: The first character known as Human Torch, he is an android...

2. [0.394] Android (disambiguation)
   Android most commonly refers to: Android (robot), a humanoid robot...
```

RAG mode (retriever + LLM):

```text
$ python3 vectorize.py -llm "what is an android"

An android is a humanoid robot or synthetic organism designed to imitate a human.
It can also refer to Google's operating system for mobile devices.
```

## Files

- loader.py - Downloads Wikipedia articles
- vectorize.py - Main script (search + optional LLM)
- chunker.py - Text chunking
- topic_expansion.py - Builds synonym dictionary
- topic_expansion_data.py - Generated synonyms
- wiki/ - Downloaded articles

## Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended for LLM)
- Ollama installed for LLM mode: curl -fsSL https://ollama.com/install.sh | sh

## License

MIT
