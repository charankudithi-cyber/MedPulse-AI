from setuptools import setup, find_packages

setup(
    name="medpulse-ai",
    version="1.0.0",
    author="Charan Kudithi",
    author_email="charankudithi@gmail.com",
    description="Intelligent RAG-Powered Healthcare Assistant using LangChain, Pinecone, and Generative AI",
    long_description_content_type="text/markdown",
    url="https://github.com/charankudithi-cyber/MedPulse-AI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=[
        "flask",
        "langchain",
        "langchain-community",
        "langchain-core",
        "langchain-pinecone",
        "langchain-openai",
        "sentence-transformers",
        "pypdf",
        "python-dotenv",
        "pinecone-client",
    ],
)
