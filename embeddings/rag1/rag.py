from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from textwrap import wrap
import pandas as pd
import numpy as np

# https://huggingface.co/datasets/MongoDB/embedded_movies
dataset = load_dataset("MongoDB/embedded_movies")
dataset_df = pd.DataFrame(dataset['train'])
print(dataset_df.info())

# Models: gte-small, gte-base, gte-large
embedding_model = SentenceTransformer("thenlper/gte-large")
counter = 0
stop_at = 10  # dataset_df.shape[0]
dataset_df = dataset_df.head(stop_at)


def get_embedding(text: str) -> list[float]:
    global counter
    counter += 1
    if counter % 10 == 0:
        print(f"Processed {counter} texts.")
    if not text or not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)
    return embedding.tolist()


dataset_df["embedding"] = dataset_df["fullplot"].apply(get_embedding)
# Print the columns fullplot and embedding for the first 5 rows
print(dataset_df[["fullplot", "embedding"]].head())


def similarity(query_embedding, text_embedding):
    return np.dot(text_embedding, query_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(query_embedding))


# Ask user for a search query
text = input("Search for movies: ")
print()
if text:
    query_embedding = get_embedding(text)

    docs_similarity = []
    for i, row in dataset_df.iterrows():
        text_embedding = row["embedding"]
        sim = similarity(query_embedding, text_embedding)
        docs_similarity.append({
            "fullplot": row["fullplot"],
            "similarity": sim
        })

    # Sort the most similar documents for docs similarity
    docs_similarity = sorted(docs_similarity, key=lambda x: x["similarity"], reverse=True)

    # Slice first 5 documents from docs_similarity
    for idx, obj in enumerate(docs_similarity[:5]):
        print(f"Doc #{idx}, similarity: {obj['similarity']}")
        # Print fullplot text breaking lines after 80 columns
        print("Fullplot:")
        lines = wrap(obj["fullplot"], 120)
        print("\n".join(lines))
        print()
