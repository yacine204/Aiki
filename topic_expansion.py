import argparse
import os
import time
from typing import Dict, List, Set

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "AikiTopicExpansion/1.0 (contact: local)"


def list_topics(directory: str = "wiki") -> List[str]:
    if not os.path.isdir(directory):
        return []
    topics = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            topics.append(filename[:-4].replace("_", " "))
    return sorted(set(topics))


def wiki_request(params: Dict) -> Dict:
    try:
        response = requests.get(
            WIKI_API,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return {}


def fetch_redirects(title: str, limit: int = 50) -> List[str]:
    params = {
        "action": "query",
        "titles": title,
        "prop": "redirects",
        "rdlimit": limit,
        "format": "json",
    }
    data = wiki_request(params)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        redirects = page.get("redirects", [])
        return [r.get("title") for r in redirects if r.get("title")]
    return []


def fetch_links(title: str, limit: int = 50) -> List[str]:
    params = {
        "action": "query",
        "titles": title,
        "prop": "links",
        "plnamespace": 0,
        "pllimit": limit,
        "format": "json",
    }
    data = wiki_request(params)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        links = page.get("links", [])
        return [l.get("title") for l in links if l.get("title")]
    return []


def fetch_opensearch(title: str, limit: int = 10) -> List[str]:
    params = {
        "action": "opensearch",
        "search": title,
        "namespace": 0,
        "limit": limit,
        "format": "json",
    }
    data = wiki_request(params)
    if isinstance(data, list) and len(data) >= 2:
        suggestions = data[1]
        return [s for s in suggestions if isinstance(s, str)]
    return []


def normalize_list(items: List[str], excluded: Set[str]) -> List[str]:
    cleaned = []
    seen = set()
    for item in items:
        if not item:
            continue
        if item in excluded:
            continue
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def expand_topic(
    title: str,
    synonym_limit: int,
    related_limit: int,
    sleep_seconds: float,
) -> List[str]:
    synonyms = fetch_redirects(title, limit=synonym_limit)
    time.sleep(sleep_seconds)
    related_from_links = fetch_links(title, limit=related_limit)
    time.sleep(sleep_seconds)
    related_from_search = fetch_opensearch(title, limit=min(related_limit, 25))

    excluded = {title}
    excluded.update(synonyms)

    synonyms = normalize_list(synonyms, {title})
    related = normalize_list(related_from_links + related_from_search, excluded)

    return normalize_list(synonyms + related, {title})


def build_topic_expansion(
    topics: List[str],
    synonym_limit: int,
    related_limit: int,
    sleep_seconds: float,
    max_topics: int,
) -> Dict[str, List[str]]:
    results = {}
    for index, topic in enumerate(topics):
        if max_topics and index >= max_topics:
            break
        results[topic] = expand_topic(
            topic,
            synonym_limit=synonym_limit,
            related_limit=related_limit,
            sleep_seconds=sleep_seconds,
        )
        print(f"[{index + 1}/{min(len(topics), max_topics or len(topics))}] {topic}")
    return results


def write_python_output(path: str, data: Dict[str, List[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("topic_expansion = ")
        f.write("{\n")
        for topic, words in data.items():
            words_literal = ", ".join([repr(w) for w in words])
            f.write(f"    {repr(topic)}: [{words_literal}],\n")
        f.write("}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate topic expansion data.")
    parser.add_argument("--wiki-dir", default="wiki", help="Directory with wiki topics.")
    parser.add_argument("--output", default="topic_expansion_data.py", help="Output Python path.")
    parser.add_argument("--synonyms", type=int, default=25, help="Synonym limit per topic.")
    parser.add_argument("--related", type=int, default=50, help="Related topic limit per topic.")
    parser.add_argument("--sleep", type=float, default=0.4, help="Sleep between requests.")
    parser.add_argument("--max-topics", type=int, default=0, help="Limit number of topics.")
    args = parser.parse_args()

    topics = list_topics(args.wiki_dir)
    if not topics:
        print("No topics found.")
        return

    results = build_topic_expansion(
        topics,
        synonym_limit=args.synonyms,
        related_limit=args.related,
        sleep_seconds=args.sleep,
        max_topics=args.max_topics,
    )

    write_python_output(args.output, results)




if __name__ == "__main__":
    main()
