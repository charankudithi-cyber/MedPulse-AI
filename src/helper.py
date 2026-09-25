import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


def load_pdf_file(data_path: str) -> List[Document]:
    """
    Extract text documents from PDF files located within a specified directory.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The data directory '{data_path}' does not exist.")

    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filter documents to retain only essential source metadata and text content
    to optimize token usage and vector upserts.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


def text_split(extracted_data: List[Document], chunk_size: int = 500, chunk_overlap: int = 20) -> List[Document]:
    """
    Split extracted documents into smaller chunks for optimal semantic embedding and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Load dense text embedding model from HuggingFace (defaults to 384-dimensional all-MiniLM-L6-v2).
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
