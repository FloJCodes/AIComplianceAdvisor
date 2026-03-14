"""
Pushes embeddings + chunks to Azure AI Search.

Reads the JSON produced by embed_database.py,
creates a search index with vector search, and uploads all documents.
"""

import json
import os

from dotenv import load_dotenv
load_dotenv()

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.core.credentials import AzureKeyCredential

INDEX_NAME = "eu-ai-act-index"
EMBEDDINGS_PATH = "data/azure_embedded_database/eu_ai_act.json"

def create_index(index_client: SearchIndexClient, embedding_dim: int):
    fields = [
        SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="text", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SearchableField(name="chapter", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="section", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dim,
            vector_search_profile_name="default-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
        profiles=[VectorSearchProfile(name="default-vector-profile", algorithm_configuration_name="default-hnsw")],
    )

    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
    result = index_client.create_or_update_index(index)


def upload_documents(search_client: SearchClient, chunks: list[dict]):
    batch_size = 100
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        documents = []

        for chunk in batch:
            doc = {
                "chunk_id": str(chunk["index"]),
                "text": chunk["text"],
                "chapter": chunk["metadata"]["chapter"],
                "section": chunk["metadata"]["section"],
                "chunk_index": chunk["metadata"]["chunk"],
                "embedding": chunk["embedding"],
            }
            documents.append(doc)

        result = search_client.upload_documents(documents=documents)


def main():
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    api_key = os.environ["AZURE_SEARCH_API_KEY"]
    credential = AzureKeyCredential(api_key)

    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]
    embedding_dim = data["dimension"]

    index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
    create_index(index_client, embedding_dim)

    search_client = SearchClient(endpoint=endpoint, index_name=INDEX_NAME, credential=credential)
    upload_documents(search_client, chunks)


if __name__ == "__main__":
    main()
