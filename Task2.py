import requests
import json
import time
import re
from bs4 import BeautifulSoup

def clean_html(html_text):
    """Strip HTML tags and extra whitespace."""
    soup = BeautifulSoup(html_text, 'html.parser')
    text = soup.get_text(separator=" ").strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text

def fetch_with_retry(url, headers=None, retries=3, delay=5, timeout=30):
    """Helper to retry API requests on failure."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt < retries:
                print(f"🔁 Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"❌ Failed after {retries} attempts.")
    return None

def collect_qa(tag="harry-potter", pages=5, output_file="qa_dataset.json"):
    all_qa = []
    headers = {
        'User-Agent': 'QADataCollectorBot/1.0'
    }

    for page in range(1, pages + 1):
        print(f"🔎 Fetching questions from page {page}...")
        url = (
            f"https://api.stackexchange.com/2.3/questions?"
            f"page={page}&pagesize=20&order=desc&sort=votes"
            f"&tagged={tag}&site=scifi&filter=withbody"
        )

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ Failed to fetch page {page}: {e}")
            continue

        for item in data.get("items", []):
            if not item.get("is_answered"):
                continue

            question = clean_html(item.get("title", ""))
            q_id = item.get("question_id")
            q_body = clean_html(item.get("body", ""))

            # Retry-enabled answer fetch
            answer_url = (
                f"https://api.stackexchange.com/2.3/questions/{q_id}/answers?"
                f"order=desc&sort=votes&site=scifi&filter=withbody"
            )
            answer_data = fetch_with_retry(answer_url, headers=headers)

            if answer_data and answer_data.get("items"):
                top_answer_html = answer_data["items"][0].get("body", "")
                top_answer = clean_html(top_answer_html)

                if top_answer:
                    all_qa.append({
                        "question": question,
                        "question_body": q_body,
                        "answer": top_answer,
                        "question_id": q_id,
                        "source": f"https://scifi.stackexchange.com/questions/{q_id}"
                    })

            time.sleep(0.5)  # polite API usage

    if not all_qa:
        print("❌ No Q&A pairs collected. Please verify API response or tag.")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saved {len(all_qa)} Q&A pairs to {output_file}")

# Run example
if __name__ == "__main__":
    collect_qa(tag="harry-potter", pages=5)
