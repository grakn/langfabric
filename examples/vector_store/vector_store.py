import argparse
import asyncio
from langchain_core.vectorstores import InMemoryVectorStore
from langfabric import load_model_configs, build_embeddings

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

    vector_store = InMemoryVectorStore.from_texts(
        docs,
        embedding=llm,
    )

    # Run similarity search with scores
    query = "What is the temperature?"
    results = vector_store.similarity_search_with_score(query)

    # Print document content and score
    for doc, score in results:
        print(f"[Score: {score:.4f}] {doc.page_content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a embedding model from langfabric")
    parser.add_argument("model_name", help="Name of the model defined in models.yaml")
    args = parser.parse_args()

    asyncio.run(main(args.model_name))
