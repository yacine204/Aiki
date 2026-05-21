# generate .txt instances of topics that'll be used as a dataset
import os
import requests
import json
import re 
import random
import time

iteration = 0
downloaded_topics = set()

def sanitize_title(title: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "_", title).strip()

def count_wiki_files() -> int:
    if not os.path.isdir("wiki"):
        return 0
    return len([f for f in os.listdir("wiki") if f.endswith(".txt")])

def fetch_random_titles(batch: int = 10):
    user_agent = {"User-agent": "Mozilla/5.0"}
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": batch,
        "format": "json",
    }
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=params,
            headers=user_agent,
            timeout=15,
        )
        response.raise_for_status()
        response_json = response.json()
    except (requests.RequestException, json.JSONDecodeError):
        return []

    random_pages = response_json.get("query", {}).get("random", [])
    return [page.get("title") for page in random_pages if page.get("title")]

def fetch_wiki(topic: str):
    global iteration
    if topic in downloaded_topics:
        return []

    user_agent = {"User-agent": "Mozilla/5.0"}
    params = {
        "action": "query",
        "titles": topic,
        "prop": "extracts|links",
        "explaintext": True,
        "plnamespace": 0,
        "pllimit": 50,
        "format": "json",
    }
    os.makedirs("wiki", exist_ok=True)
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=params,
            headers=user_agent,
            timeout=15,
        )
        response.raise_for_status()
        response_json = response.json()
    except (requests.RequestException, json.JSONDecodeError):
        return []

    pages = response_json.get("query", {}).get("pages", {})

    for page in pages.values():
        if "extract" in page:
            extract_text = page["extract"]
            safe_title = sanitize_title(topic)
            file_path = f"wiki/{safe_title}.txt"
            if not os.path.exists(file_path):
                with open(file_path, "w+", encoding="utf-8") as file:
                    file.write(extract_text)
                downloaded_topics.add(topic)
                iteration += 1
                print(f"[{iteration}] saved: {topic}")

            see_also_pattern = r"^==+\s*See also\s*==+\s*\n(.*?)(?=\n==+\s*[^=]+\s*==+\s*\n|\Z)"
            match = re.search(see_also_pattern, extract_text, re.DOTALL | re.MULTILINE)
            topics = []
            if match:
                see_also_text = match.group(1)
                topics = [line.strip("* ") for line in see_also_text.splitlines() if line.strip()]
            if not topics:
                links = page.get("links", [])
                topics = [link.get("title") for link in links if link.get("title")]
            return [t for t in topics if t]
    return []

def get_wiki(seed: str, limit: int = 150, sample_size: int = 5):
    global iteration
    iteration = count_wiki_files()
    if os.path.isdir("wiki"):
        for filename in os.listdir("wiki"):
            if filename.endswith(".txt"):
                downloaded_topics.add(os.path.splitext(filename)[0].replace("_", " "))
    queue = [seed]
    while iteration < limit:
        if not queue:
            queue.extend(fetch_random_titles())
            if not queue:
                time.sleep(1)
                continue
        topic = queue.pop(0)
        if topic in downloaded_topics:
            continue
        next_topics = fetch_wiki(topic)
        if next_topics:
            filtered = [t for t in next_topics if t not in downloaded_topics]
            queue.extend(filtered[:sample_size])

get_wiki("Artificial intelligence")

# TODO : implement threads instead of queus