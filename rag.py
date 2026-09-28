from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# Paths
KNOWLEDGE_FILE = Path("data/knowledge/crop_knowledge.txt")
INDEX_FILE = Path("models/agriculture.index")
TEXT_FILE = Path("models/agriculture_chunks.txt")


# Load embedding model
print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Load FAISS index
index = faiss.read_index(str(INDEX_FILE))


# Load knowledge chunks
chunks = TEXT_FILE.read_text(
    encoding="utf-8"
).split("\n\n")


def search_knowledge(query, top_k=3):
    """
    Search the agricultural knowledge base.
    """

    # Convert query into embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):

        if index_id == -1:
            continue

        results.append({
            "score": float(score),
            "text": chunks[index_id]
        })

    return results


# Test query
query = "What are the water and nutrient requirements for rice?"

results = search_knowledge(query)


print("\n🔎 Search Query:")
print(query)

print("\n📚 Retrieved Knowledge:")
print("------------------------")

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print(f"Similarity Score: {result['score']:.4f}")
    print(result["text"])