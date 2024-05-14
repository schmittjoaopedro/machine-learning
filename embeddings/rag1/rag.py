from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from textwrap import wrap
import pandas as pd
import numpy as np

# Models: gte-small, gte-base, gte-large
embedding_model = SentenceTransformer("thenlper/gte-large")


def get_embedding(content: str) -> list[float]:
    if not content or not content.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(content)
    return embedding.tolist()


def calculate_similarity(query_emb, response_emb):
    if not response_emb:
        similarity = 0
        distance = 1
    else:
        similarity = np.dot(response_emb, query_emb) / (np.linalg.norm(response_emb) * np.linalg.norm(query_emb))
        distance = np.linalg.norm(np.array(query_emb) - np.array(response_emb))
    return similarity, distance


# If pre-processed CSV file already exists
try:
    dataset_df = pd.read_csv("movies.csv")
    dataset_df["embedding"] = dataset_df["embedding"].apply(eval)
    # Ignore documents with empty embeddings
    dataset_df = dataset_df[dataset_df["embedding"].apply(lambda x: bool(x))]
except FileNotFoundError:
    dataset_df = pd.DataFrame()

if dataset_df.empty:
    # https://huggingface.co/datasets/MongoDB/embedded_movies
    dataset = load_dataset("MongoDB/embedded_movies")
    dataset_df = pd.DataFrame(dataset['train'])
    print(dataset_df.info())

    dataset_df["embedding"] = dataset_df["fullplot"].apply(get_embedding)
    dataset_df = dataset_df[["fullplot", "embedding"]]
    print(dataset_df.head())

    # Save the dataset to a CSV file
    dataset_df.to_csv("movies.csv", index=False)

# Ask user for a search query
text = input("Search for movies: ")
print()
if text:
    query_embedding = get_embedding(text)

    docs_similarity = []
    for i, row in dataset_df.iterrows():
        text_embedding = row["embedding"]
        sim, dist = calculate_similarity(query_embedding, text_embedding)
        docs_similarity.append({
            "fullplot": row["fullplot"],
            "similarity": sim,
            "distance": dist,
        })

    docs_similarity = sorted(docs_similarity, key=lambda x: x["similarity"], reverse=True)
    print("Top 5 most similar documents:")
    for idx, obj in enumerate(docs_similarity[:5]):
        print(f"Doc #{idx}, similarity: {obj['similarity']}, distance: {obj['distance']}")
        # Print fullplot text breaking lines after 80 columns
        print("Fullplot:")
        lines = wrap(obj["fullplot"], 160)
        print("\n".join(lines))
        print()
