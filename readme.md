# Aiki

Wikipedia search engine using TF-IDF and cosine similarity. Built from scratch with NumPy.

## What it does

- Downloads Wikipedia articles into wiki/
- Splits documents into 500-character chunks
- Builds TF-IDF vectors (vocabulary size ~21k)
- Searches using cosine similarity
- Supports query expansion via Wikipedia redirects/links

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install requests numpy langchain-text-splitters
```

## Usage

Download articles:

```bash
python3 loader.py
```

Search (default query: "android"):

```bash
python3 vectorize.py "what is an android"
```

Generate topic expansion data:

```bash
python3 topic_expansion.py --wiki-dir wiki
```

## Output example

```text
$ python3 vectorize.py "android"

1. [0.401] List of fictional robots and androids
	Human Torch: The first character known as Human Torch, he is an android...

2. [0.394] Android (disambiguation)
	Android most commonly refers to: Android (robot), a humanoid robot...

3. [0.389] Artificial intelligence
	A related concept focusing on machine learning...
```

## Files

- loader.py - Downloads Wikipedia articles
- vectorize.py - Main search script
- chunker.py - Text chunking
- topic_expansion.py - Builds synonym dictionary
- topic_expansion_data.py - Generated synonyms
- wiki/ - Downloaded articles

## License

MIT
