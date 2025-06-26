import argparse
import asyncio
from langfabric import load_model_configs, build_embeddings
import numpy as np

async def main(model_name: str):
    # Load model configs
    cfg = load_model_configs(["models.yaml"])

    if model_name not in cfg:
        raise ValueError(f"Model '{model_name}' not found in config")

    # Build the selected model
    llm = build_embeddings(cfg[model_name])

    docs = [
        "What is the weather temperature today?",
        "What is the temperature of the model?"
    ]
    doc_vectors = llm.embed_documents(docs)
    query_vector = llm.embed_query("What is the temperature?")

    doc_vectors = np.array(doc_vectors)
    query_vector = np.array(query_vector)

    # Normalize vectors (optional but recommended for cosine similarity)
    doc_norms = np.linalg.norm(doc_vectors, axis=1)
    query_norm = np.linalg.norm(query_vector)

    # Compute cosine similarity
    similarities = (doc_vectors @ query_vector) / (doc_norms * query_norm)

    # Print similarity scores
    for i, score in enumerate(similarities):
        print(f"[Score: {score:.4f}] {docs[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a embedding model from langfabric")
    parser.add_argument("model_name", help="Name of the model defined in models.yaml")
    args = parser.parse_args()

    asyncio.run(main(args.model_name))
