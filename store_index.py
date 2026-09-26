import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, filter_to_minimal_docs, download_hugging_face_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medpulse-ai")


def index_medical_knowledge_base():
    """
    Ingests medical documents from data/, chunks the content, computes embeddings,
    and indexes them into a Pinecone serverless vector database.
    """
    if not PINECONE_API_KEY or PINECONE_API_KEY.strip() == "" or "your_" in PINECONE_API_KEY:
        print("[!] Error: PINECONE_API_KEY is not configured. Please set it in your .env file.")
        sys.exit(1)

    data_dir = "data"
    print(f"[*] Inspecting medical data directory: '{data_dir}'...")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"[!] Created directory '{data_dir}'. Add medical reference PDFs to index.")
        return

    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[!] No PDF documents found in '{data_dir}'. Please place medical PDFs (e.g. Gale Encyclopedia) into the data/ folder.")
        return

    print(f"[*] Found {len(pdf_files)} document(s): {', '.join(pdf_files)}")
    print("[*] Extracting pages from PDF files...")
    extracted_data = load_pdf_file(data_dir)
    print(f"[+] Successfully extracted {len(extracted_data)} pages.")

    print("[*] Filtering document metadata and splitting into semantic chunks...")
    minimal_docs = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(minimal_docs, chunk_size=500, chunk_overlap=20)
    print(f"[+] Generated {len(text_chunks)} text chunks.")

    print("[*] Downloading HuggingFace sentence-transformers embedding model...")
    embeddings = download_hugging_face_embeddings()

    print("[*] Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"[*] Index '{INDEX_NAME}' not found. Creating serverless index (dimension=384, metric=cosine)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"[+] Pinecone index '{INDEX_NAME}' created successfully.")
    else:
        print(f"[*] Pinecone index '{INDEX_NAME}' verified.")

    print(f"[*] Upserting {len(text_chunks)} chunks to index '{INDEX_NAME}'...")
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print("[+] Ingestion complete! Knowledge base is ready for RAG queries.")


if __name__ == "__main__":
    index_medical_knowledge_base()
