import argparse
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langfabric import load_model_configs, build_model


async def main(model_name: str):
    # Load model configs
    cfg = load_model_configs(["models.yaml"])

    if model_name not in cfg:
        raise ValueError(f"Model '{model_name}' not found in config")

    # Build the selected model
    llm = build_model(cfg[model_name])

    # Define a prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant answering questions about the region {region_name}. Provide short and clear answers.",
        ),
        ("human", "{input}"),
    ])

    # Run the chain
    chain = prompt | llm
    output = await chain.ainvoke({
        "region_name": "Bay Area",
        "input": "How many people are living there?",
    })

    print(output.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a chat model from langfabric")
    parser.add_argument("model_name", help="Name of the model defined in models.yaml")
    args = parser.parse_args()

    asyncio.run(main(args.model_name))
