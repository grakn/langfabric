from setuptools import setup, find_packages

setup(
    name="langfabric",
    version="0.1.5",
    description="Flexible LLM model initialization from YAML configuration.",
    author="Phragman",
    author_email="a@staphi.com",
    url="https://github.com/grakn/langfabric",
    license="MIT",
    packages=find_packages(include=["langfabric", "langfabric.*"]),
    install_requires=[
        "pydantic>=2.5,<3.0",
        "pyyaml>=6.0,<7.0",
        "seyaml>=0.1.1,<1.0",
        "langchain>=0.3.26,<1.0",
        "langchain-openai>=0.3.24,<1.0",
        "langchain-community>=0.3.26,<1.0",
        "langchain-groq>=0.3.2,<1.0",
    ],
    python_requires=">=3.9,<4.0",
    extras_require={
        "dev": [
            "pytest>=8.0,<9.0",
            "pytest-cov>=4.0,<5.0",
            "mypy>=1.0,<2.0",
            "ruff>=0.1.0,<0.5.0",  # Optional but helpful for linting
        ],
        "docs": [
            "mkdocs>=1.5,<2.0",
            "mkdocs-material>=9.0,<10.0",
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
