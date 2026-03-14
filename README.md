# AIComplianceAdvisor

A RAG (Retrieval-Augmented Generation) system that answers questions about the **EU Artificial Intelligence Act** (Regulation (EU) 2024/1689) grounded in the actual legal text, with source citations.


## Project Structure

```
AIComplianceAdvisor/
|── src/
│   |── parse_eu_ai_act.py       # Parse EUR-Lex HTML → structured text files
│   |── embed_database.py        # Chunk + embed (local or Azure)
│   |── search_index.py          # Create Azure AI Search index + upload
│   |── rag_chain.py             # Retrieve + generate (RAG vs Direct)
├── data/
│   |── eu_ai_act.html           # Saved from EUR-Lex
│   |── eu_ai_act/               # Parsed text files
│   │   |── recitals/            # 179 Recitals
│   │   |── Chapter_I_.../       # Articles grouped by Chapter
│   │   |── ...
│   │   |── annexes/             # 13 Annexes
│   |── local_embedded_database/ # Embeddings (all-MiniLM-L6-v2, dimension 384)
│   |── azure_embedded_database/ # Embeddings (ada-002, dimension 1536)
|── docs/
│   |── Answer_Examples.txt      # RAG vs Direct answer comparisons
|── requirements.txt
|── .env.example
|── README.md
```


## How It Works

1.  The EU AI Act HTML is parsed into structured text files — 179 recitals, 114 articles across 13 chapters, and 13 annexes
2.  Each file is chunked into 500 word pieces with overlap to preserve context
3.  Chunks are embedded using local model `all-MiniLM-L6-v2` (dimension 384) or Azure OpenAI `text-embedding-ada-002` (dimension 1536)
4.  Azure OpenAI Embeddings are indexed in Azure AI Search with hybrid search (vector similarity + keyword matching)
5.  When a question is asked, it's embedded and the 5 most relevant chunks are retrieved
6.  Retrieved chunks + the question are sent to Claude, which generates an answer citing specific articles

## Why RAG Matters

The system includes a comparison mode that shows the same question answered with and without retrieved context. Without RAG, even strong models like Claude hallucinate on legal details:

**Question:** _"What are the 12 elements that must be included in the EU declaration of conformity according to Annex V?"_


| Syntax | RAG System | Claude |
| --- | ----------- | --------- |
| Count | Correctly identifies **8 elements**, states there are not 12 | Lists 12 as requested, inventing duplicates |
| Accuracy | All 8 match the actual Annex V text | Splits single elements into multiple to reach 12 |
| Hallucination | None — says "the context does not contain 12 elements" | Points 7 & 12 are both "additional information" |

On Broad questions both models answer similarly. The strength of the RAG System lies in its accuracy on specific questions about the AI Act. It doesnt hallucinate and tells you when the contexst doesnt contain anything mentioned in your question.
Full examples can be found in `docs/Answer_Examples.txt`.

## Setup

### Prerequisites

-   Python 3.10+
-   Azure account with Azure OpenAI and Azure AI Search
-   Anthropic API key

### Installation

```bash
git clone https://github.com/FlojCodes/AIComplianceAdvisor.git
cd AIComplianceAdvisor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Fill in  `.env`:

```
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=your-key

AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002

ANTHROPIC_API_KEY=sk-ant-your-key

```
