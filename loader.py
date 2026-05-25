# generate .txt instances of topics that'll be used as a dataset
# use : 
# python3 loader.py "Interest" --type (Topic/Category) --l (number of docs) --ss (number of relawantedted docs) 

import os
import requests
import json
import re 
import random
import time
import argparse

iteration = 0
downloaded_topics = set()

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "AikiDatasetBot/1.0 (contact: local)"
MIN_REQUEST_INTERVAL = 0.35
_last_request_time = 0.0

def request_with_backoff(
    params: dict,
    max_retries: int = 4,
    base_sleep: float = 0.75,
    max_sleep: float = 5.0,
):
    global _last_request_time
    for attempt in range(max_retries):
        try:
            now = time.time()
            elapsed = now - _last_request_time
            if elapsed < MIN_REQUEST_INTERVAL:
                time.sleep(MIN_REQUEST_INTERVAL - elapsed)
            response = requests.get(
                WIKI_API_URL,
                params=params,
                headers={"User-agent": USER_AGENT},
                timeout=15,
            )
            _last_request_time = time.time()
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_time = float(retry_after)
                else:
                    sleep_time = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(min(max_sleep, sleep_time))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException:
            sleep_time = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(min(max_sleep, sleep_time))
    return None

def sanitize_title(title: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "_", title).strip()

def count_wiki_files() -> int:
    if not os.path.isdir("wiki"):
        return 0
    return len([f for f in os.listdir("wiki") if f.endswith(".txt")])

def fetch_random_titles(batch: int = 10):
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": batch,
        "format": "json",
    }
    try:
        response = request_with_backoff(params)
        if response is None:
            return []
        response_json = response.json()
    except (requests.RequestException, json.JSONDecodeError):
        return []

    random_pages = response_json.get("query", {}).get("random", [])
    return [page.get("title") for page in random_pages if page.get("title")]

def fetch_wiki_by_topic(topic: str):
    global iteration
    if topic in downloaded_topics:
        return []

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
        response = request_with_backoff(params)
        if response is None:
            return []
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

def find_category_title(category: str):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": category,
        "srnamespace": 14,
        "srlimit": 1,
        "format": "json",
    }
    response = request_with_backoff(params)
    if response is None:
        return None
    try:
        data = response.json()
    except json.JSONDecodeError:
        return None
    results = data.get("query", {}).get("search", [])
    if not results:
        return None
    title = results[0].get("title", "")
    if title.startswith("Category:"):
        return title.replace("Category:", "", 1)
    return title or None

def fetch_wiki_by_category(category: str, limit: int):
    all_topics = []
    continue_token = None
    resolved_category = category
    attempted_resolution = False
    while True:
        remaining = max(0, limit - len(all_topics))
        if remaining == 0:
            break
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{resolved_category}",
            "cmtype": "page",
            "cmlimit": min(remaining, 500),
            "format": "json",
        }
        
        if continue_token:
            params["cmcontinue"] = continue_token
        
        response = request_with_backoff(params)
        if response is None:
            if not all_topics:
                print("No response from Wikipedia API. Try again later or reduce --l.")
            break
        data = response.json()
        if "error" in data:
            print(f"Wikipedia API error: {data['error'].get('info', 'unknown error')}")
            break
        
        topics = data.get("query", {}).get("categorymembers", [])
        if not topics and not all_topics:
            if not attempted_resolution:
                attempted_resolution = True
                resolved = find_category_title(category)
                if resolved and resolved != resolved_category:
                    resolved_category = resolved
                    print(f"Resolved category to: {resolved_category}")
                    continue
            print("No pages found for this category. Check the category name.")
        for topic in topics: 
            print(topic["title"])
        batch_titles = [topic["title"] for topic in topics]
        all_topics.extend(batch_titles)
        for title in batch_titles:
            fetch_wiki_by_topic(topic=title)
        continue_token = data.get("continue", {}).get("cmcontinue")
        if not continue_token:
            break
    
    return all_topics

seed_type_type = ["Category" ,"Topic"]
def get_wiki(seed: str, seed_type: str , limit: int, sample_size: int):
    global iteration
    if seed_type not in seed_type_type:
        print("Invalid seed_type, must be Category or Topic")
        return
    
    iteration = count_wiki_files()
    target_total = iteration + limit
    if os.path.isdir("wiki"):
        for filename in os.listdir("wiki"):
            if filename.endswith(".txt"):
                downloaded_topics.add(os.path.splitext(filename)[0].replace("_", " "))
    if seed_type == "Category":
        fetch_wiki_by_category(seed, limit)
        return

    queue = [seed]
    while iteration < target_total:
        if not queue:
            queue.extend(fetch_random_titles())
            if not queue:
                time.sleep(1)
                continue
        topic = queue.pop(0)
        if topic in downloaded_topics:
            continue
        next_topics = fetch_wiki_by_topic(seed)

        if next_topics:
            filtered = [t for t in next_topics if t not in downloaded_topics]
            queue.extend(filtered[:sample_size])


if __name__  == "__main__":

    # ways of fetching: by_category, by_topic

    parser = argparse.ArgumentParser(description='Wikipedia scrapper, include your intereseted topic/category and it will fetch related topics too within the limits you include')

    parser.add_argument("topic", nargs="+", help = "First topic in the scrapped chain")
    parser.add_argument("--type", help = "input type (topic/category)")
    parser.add_argument("--l", help= "Number of elements in the scrapped chain")
    parser.add_argument("--ss", help= "Number of related topics non downloaded yet in the queue")

    args = parser.parse_args()
    
    if not args.topic or not args.type or not args.l or not args.ss :
        print("missing parameters, must include --topic, --l, --ss.\n...python3 loader.py --topic Machine learning --l 150 --ss 5")
    else:
        topic = " ".join(args.topic)
        get_wiki(topic, args.type, int(args.l), int(args.ss))
            