import os
import openai
import pinecone
from langchain.embeddings import OpenAIEmbeddings

query = "What is Antler?"
embeddings_model = OpenAIEmbeddings(
    openai_api_key="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"
)
embedded_query = embeddings_model.embed_query(query)
print(embedded_query[:5])


pinecone.init(
    api_key="3405e0f4-766b-41a4-a69f-59ad4c6c4af7", environment="us-west4-gcp-free"
)
# openai.organization = "openai-org-id"
openai.api_key = "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"


response = openai.Embedding.create(input=query, model="text-embedding-ada-002")
query_embedding = response["data"][0]["embedding"]

num_results = 5
index = pinecone.Index("index1536")


try:
    results = index.query(embedded_query, k=num_results, top_k=5, include_metadata=True)
except Exception as e:
    print(f"Error during querying: {e}")


print(results)
for match in results["matches"]:
    print(f"Text chunk: {match['metadata']['text']}")
    print(f"Similarity score: {match['score']}")
    print("\n")
