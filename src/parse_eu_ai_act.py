"""
Parses the EU AI Act from a locally saved EUR-Lex HTML file.

Converts HTML to plain text, then splits into recitals, articles, and annexes by regex.
Articles are grouped into subdirectories by chapter.
"""

import os
import re

from bs4 import BeautifulSoup

HTML_PATH = "data/eu_ai_act.html"
OUTPUT_DIR = "data/eu_ai_act"


def clean_filename(text: str) -> str:
    text = re.sub(r'[\xa0\u200b\\/*?:"<>|,;.\n\r\t]', '_', text)
    text = text.replace("—", "_").replace("–", "_").replace(" ", "_")
    return re.sub(r'_+', '_', text).strip('_')[:80]


def clean_text(text: str) -> str:
    text = text.replace('\xa0', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = text.replace('\xa0', ' ')
    return clean_text(text)


def save_section(directory: str, filename: str, text: str):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename + ".txt"), "w", encoding="utf-8") as f:
        f.write(text)


def extract_recitals(full_text: str) -> int:
    match = re.search(r'Whereas:\s*\n(.*?)HAVE ADOPTED THIS REGULATION', full_text, re.DOTALL)
    if not match:
        return 0

    parts = re.split(r'\n\((\d+)\)\s*', match.group(1))
    count = 0
    for i in range(1, len(parts) - 1, 2):
        text = clean_text(parts[i + 1])
        if text:
            save_section(os.path.join(OUTPUT_DIR, "recitals"), f"recital_{parts[i]}", text)
            count += 1
    return count


def extract_articles_and_annexes(full_text: str) -> int:
    match = re.search(r'HAVE ADOPTED THIS REGULATION:\s*\n(.*)', full_text, re.DOTALL)
    if not match:
        return 0

    body = match.group(1)
    article_splits = re.split(r'\n(Article \d+)\s*\n', body)

    current_dir = os.path.join(OUTPUT_DIR, "ungrouped")
    count = 0

    for i in range(1, len(article_splits) - 1, 2):
        header = article_splits[i].strip()
        content = article_splits[i + 1]
        preceding = article_splits[i - 1]

        # Detect chapter boundary in the text before this article
        chapters = re.findall(r'CHAPTER ([IVX]+)\n([A-Z][A-Z\s,\-]+?)(?=\n|\Z)', preceding)
        if chapters:
            ch_num, ch_title = chapters[-1]
            current_dir = os.path.join(OUTPUT_DIR, f"Chapter_{ch_num}_{clean_filename(ch_title)}")

        # First line is the article title, rest is body
        lines = content.strip().split('\n')
        title = lines[0].strip() if lines else ""
        body_lines = [l for l in lines[1:] if not re.match(r'^ANNEX [IVX]+', l)]

        num = re.search(r'\d+', header).group()
        filename = f"Article_{num}_{clean_filename(title)[:50]}" if title else f"Article_{num}"
        save_section(current_dir, filename, clean_text('\n'.join(body_lines)))
        count += 1

    # Annexes
    annex_splits = re.split(r'\n(ANNEX [IVX]+)\s*\n', body)
    annexes_dir = os.path.join(OUTPUT_DIR, "annexes")

    for i in range(1, len(annex_splits) - 1, 2):
        lines = annex_splits[i + 1].strip().split('\n')
        roman = annex_splits[i].split()[-1]
        title = clean_filename(lines[0])[:50] if lines else ""
        text = clean_text('\n'.join(lines[1:])) if len(lines) > 1 else ""

        save_section(annexes_dir, f"Annex_{roman}_{title}" if title else f"Annex_{roman}", text)
        count += 1

    return count

def main():
    if not os.path.exists(HTML_PATH):
        print(f"HTML file not found: {HTML_PATH}")
        return

    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    full_text = html_to_text(html)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    recitals = extract_recitals(full_text)
    articles = extract_articles_and_annexes(full_text)


if __name__ == "__main__":
    main()
