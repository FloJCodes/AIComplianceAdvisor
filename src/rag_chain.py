"""
Retrieves from Azure AI Search, generates with Claude.

- Embeddings: Azure OpenAI (text-embedding-ada-002)
- Search: Azure AI Search (hybrid search)
- Generation: Anthropic Claude (claude-sonnet-4-20250514)

Usage:
  python src/rag_chain.py "<Question>"
"""

import os
import sys

from dotenv import load_dotenv
load_dotenv()

import anthropic
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

INDEX_NAME = "eu-ai-act-index"

SYSTEM_PROMPT = """You are an EU AI Act compliance assistant. You answer questions about
the EU Artificial Intelligence Act (Regulation (EU) 2024/1689) based strictly on the
provided context from the regulation.

Rules:
- Only answer based on the provided context. If the context doesn't contain enough
  information, say so clearly.
- Always cite your sources at the end.
- Be precise and use legal language where appropriate.
- If a question is ambiguous, explain different possible interpretations.
- Structure your answers clearly with the most relevant information first.
"""


class RAGChain:
    def __init__(self):
        # Azure OpenAI for embeddings
        self.embedding_client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-01",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        self.embedding_deployment = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

        # Anthropic Claude for generation
        self.llm_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.llm_model = "claude-sonnet-4-20250514"

        # Azure AI Search for retrieval
        self.search_client = SearchClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            index_name=INDEX_NAME,
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
        )

    def embed_query(self, query: str) -> list[float]:
        response = self.embedding_client.embeddings.create(
            input=[query],
            model=self.embedding_deployment,
        )
        return response.data[0].embedding

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embed_query(query)

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )

        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top_k,
            select=["chunk_id", "text", "chapter", "section", "chunk_index"],
        )

        retrieved = []
        for result in results:
            retrieved.append({
                "text": result["text"],
                "chapter": result.get("chapter", ""),
                "section": result.get("section", ""),
                "score": result["@search.score"],
            })

        return retrieved

    def generate(self, query: str, context_docs: list[dict]) -> dict:
        context_parts = []
        for doc in context_docs:
            label = doc["section"].replace("_", " ")
            context_parts.append(f"--- {label} ({doc['chapter']}) ---\n{doc['text']}")
        context_str = "\n\n".join(context_parts)

        user_message = f"""Context from the EU AI Act:

{context_str}

---
Question: {query}

Provide a detailed answer based on the context above. Cite specific articles, recitals,
or annexes in your answer."""

        response = self.llm_client.messages.create(
            model=self.llm_model,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            temperature=0.1,
            max_tokens=1500,
        )

        return {
            "answer": response.content[0].text,
            "sources": [
                {"section": doc["section"], "chapter": doc["chapter"]}
                for doc in context_docs
            ],
        }

    def query(self, question: str, top_k: int = 5, use_rag: bool = True) -> dict:
        if use_rag:
            context_docs = self.retrieve(question, top_k=top_k)
            return self.generate(question, context_docs)
        else:
            response = self.llm_client.messages.create(
                model=self.llm_model,
                system="You are an expert on the EU AI Act.",
                messages=[{"role": "user", "content": question}],
                temperature=0.1,
                max_tokens=1500,
            )
            return {
                "answer": response.content[0].text,
                "sources": [],
            }


TEST_QUESTIONS = [
    "What AI practices are completely prohibited under the EU AI Act?",
    "What are the specific fines in euros for non-compliance with prohibited AI practices vs. other violations?",
    "What are the 12 elements that must be included in the EU declaration of conformity according to Annex V?",
]


def main():
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = TEST_QUESTIONS

    chain = RAGChain()

    for question in questions:
        print(f"QUESTION: {question}")

        print(f"\n--- RAG ANSWER ---\n")
        rag_result = chain.query(question, use_rag=True)
        print(rag_result["answer"])
        print(f"\nSources:")
        for src in rag_result["sources"]:
            print(f"  - {src['section']} ({src['chapter']})")

        print(f"\n--- DIRECT ANSWER ---\n")
        direct_result = chain.query(question, use_rag=False)
        print(direct_result["answer"])

        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
