import requests
from bs4 import BeautifulSoup
import json
import time
import re

def clean_text(text):
    """Remove footnotes, brackets, and extra whitespace."""
    text = re.sub(r'\[\d+\]', '', text)  # Remove references like [1]
    return ' '.join(text.split())  # Normalize spaces and newlines

def scrape_fan_wiki(base_url, article_titles, output_file="wiki_lore_dataset.json"):
    dataset = []

    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; DataCollectorBot/1.0)'
    }

    for title in article_titles:
        url = f"{base_url}{title}"
        print(f"Scraping: {url}")
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"⚠️ Failed to fetch {url} (Status: {response.status_code})")
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.find("div", class_="mw-parser-output")
            if not content_div:
                print(f"⚠️ No main content found in {url}")
                continue

            paragraphs = content_div.find_all(["p", "h2", "h3"], recursive=False)
            lore_text = ""
            for tag in paragraphs:
                # Stop if we reach a references or non-content section
                if tag.name.startswith("h") and "references" in tag.get_text(strip=True).lower():
                    break
                if tag.name == "p":
                    text = tag.get_text().strip()
                    if text:
                        lore_text += clean_text(text) + " "

            if lore_text.strip():
                dataset.append({
                    "title": title.replace("_", " "),
                    "url": url,
                    "content": lore_text.strip()
                })

            time.sleep(1)  # Polite delay

        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")

    if not dataset:
        print("❌ No data was scraped. Please check the input URLs or network connection.")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Scraped {len(dataset)} articles into {output_file}")

# Example Usage
if __name__ == "__main__":
    base_url = "https://harrypotter.fandom.com/wiki/"
    article_titles = [
        "Harry_Potter", "Hermione_Granger", "Ron_Weasley",
        "Albus_Dumbledore", "Severus_Snape", "Hogwarts", "Potions", "Death_Eater",
        "Minerva_McGonagall", "Draco_Malfoy", "Slytherin", "Griffindor", "Horcrux"
    ]
    scrape_fan_wiki(base_url, article_titles)
