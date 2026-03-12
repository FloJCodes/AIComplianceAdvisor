"""
Chunks and embeds the scraped EU AI Act text files.

Reads the directory structure created by scrape_eu_ai_act.py,
chunks each file, generates embeddings, and saves the database.

Two modes:
  --mode local   → sentence-transformers (free, offline)
  --mode azure   → Azure OpenAI text-embedding-ada-002
"""

import os
import re
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def flatten(lst):
    #Flatten nested lists
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> list[str]:
    #Chunk text by sentence with overlap to keep context
    #Splits on newlines, then groups them into chunks that don't exceed max_tokens words

    sentences = text.split("\n")
    sentences = [s.strip() for s in sentences if s.strip()]
    buffer = []
    chunks = []

    for i, sentence in enumerate(sentences):
        words = sentence.split()

        if len(words) > max_tokens:
            # Sentence too long — fall back to word-level splitting
            if buffer:
                chunks.append(" ".join(flatten(buffer)))
                buffer = []
            # Split long sentence into sub-chunks
            for j in range(0, len(words), max_tokens - overlap):
                sub = " ".join(words[j:j + max_tokens])
                chunks.append(sub)
            continue

        if len(flatten(buffer)) + len(words) > max_tokens:
            chunks.append(" ".join(flatten(buffer)))
            # Keep overlap from end of previous chunk
            if overlap:
                buffer = flatten(buffer[-overlap:])
            else:
                buffer = []

            if len(buffer) + len(words) <= max_tokens:
                buffer.append(words)
                buffer = flatten(buffer)
            else:
                # Still too long with overlap -> overlap gets own chunk
                chunks.append(" ".join(buffer))
                buffer = words
        else:
            buffer.append(words)
            buffer = flatten(buffer)

    if buffer:
        chunks.append(" ".join(flatten(buffer)))

    return chunks


def get_embedder(mode: str):
    # Return an embedding function based on the mode
    if mode == "local":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        dimension = 384

        def embed(texts):
            return [emb.tolist() for emb in model.encode(texts, show_progress_bar=True, batch_size=32)]

        return embed, "all-MiniLM-L6-v2", dimension

    elif mode == "azure":
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-01",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        deployment = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        dimension = 1536

        def embed(texts):
            all_embs = []
            for i in range(0, len(texts), 16):
                batch = texts[i:i+16]
                resp = client.embeddings.create(input=batch, model=deployment)
                all_embs.extend([item.embedding for item in resp.data])
            return all_embs

        return embed, deployment, dimension


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "azure"], default="local")
    args = parser.parse_args()

    base_folder = os.path.join(os.getcwd(), "data", "eu_ai_act")
    output_path = os.path.join("data", f"{args.mode}_embedded_database", "eu_ai_act.json")

    embed_fn, model_name, dimension = get_embedder(args.mode)

    db = []
    index = 0
    all_texts = []
    all_metadata = []

    # Walk the directory structure
    for chapter_folder in sorted(Path(base_folder).iterdir()):
        if chapter_folder.is_dir():
            chapter_name = chapter_folder.name

            for file in sorted(chapter_folder.glob("*.txt")):
                section_text = file.read_text(encoding="utf-8")
                section_name = file.stem

                if not section_text.strip():
                    continue

                chunks = chunk_text(section_text)

                for i, chunk in enumerate(chunks):
                    chunk_clean = clean_text(chunk)
                    if not chunk_clean:
                        continue

                    all_texts.append(chunk_clean)
                    all_metadata.append({
                        "index": index,
                        "text": chunk_clean,
                        "metadata": {
                            "title": "EU AI Act (Regulation 2024/1689)",
                            "chapter": chapter_name,
                            "section": section_name,
                            "chunk": i,
                        }
                    })
                    index += 1

    # Generate embeddings
    embeddings = embed_fn(all_texts)

    # Attach embeddings
    for entry, emb in zip(all_metadata, embeddings):
        entry["embedding"] = emb

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "model": model_name,
        "dimension": dimension,
        "source": "EU AI Act (Regulation (EU) 2024/1689)",
        "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689",
        "num_chunks": len(all_metadata),
        "chunks": all_metadata,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
