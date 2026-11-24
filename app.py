
"""Streamlit frontend for DocuMind AI."""

import streamlit as st

from pathlib import Path
from hashlib import sha256

from rag_engine import (
    load_models,
    process_pdf,
    retrieve_documents,
    prepare_sources,
    format_context,
    create_chain
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "Intro_to_AI_Transcript.pdf"

MAX_FILE_SIZE = 20 * 1024 * 1024


# ==========================================
# SESSION STATE
# ==========================================

defaults = {
    "messages": [],
    "document_key": None,
    "vectorstore": None,
    "pages": 0,
    "chunks": 0
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# CACHE AI MODELS
# ==========================================

@st.cache_resource
def initialize_models():

    return load_models()


# ==========================================
# DISPLAY SOURCE PASSAGES
# ==========================================

def show_sources(sources):

    if sources:

        with st.expander(
            "📚 Retrieved Source Passages"
        ):

            for source in sources:

                st.markdown(
                    f"**Page {source['page']}**"
                )

                st.text(source["text"])
                st.divider()


# ==========================================
# MAIN HEADER
# ==========================================

st.title("🧠 DocuMind AI")

st.caption(
    "Local AI-powered document intelligence"
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🧠 DocuMind AI")

    st.caption("Private. Local. Intelligent.")

    st.divider()

    st.subheader("📂 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    st.caption("Text-based PDFs up to 20 MB.")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []

    st.subheader("Technology Stack")

    st.markdown("""
- **LLM:** Qwen 2.5 3B
- **Inference:** Ollama
- **Embeddings:** MiniLM-L6-v2
- **Vector DB:** ChromaDB
- **Framework:** LangChain
- **Frontend:** Streamlit
- **Retrieval:** Semantic + Exact Match
""")

    st.success("Local AI | No paid API")


# ==========================================
# SELECT DOCUMENT
# ==========================================

if uploaded_file is not None:

    if uploaded_file.size > MAX_FILE_SIZE:

        st.error(
            "Please upload a PDF no larger than 20 MB."
        )

        st.stop()

    pdf_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name

elif DEFAULT_PDF.exists():

    pdf_bytes = DEFAULT_PDF.read_bytes()
    filename = DEFAULT_PDF.name

else:

    st.info(
        "Upload a PDF from the sidebar to get started."
    )

    st.stop()


# ==========================================
# DOCUMENT CHANGE DETECTION
# ==========================================

document_key = (
    filename,
    sha256(pdf_bytes).hexdigest()
)

if st.session_state.document_key != document_key:

    old_store = st.session_state.vectorstore

    if old_store is not None:

        try:
            old_store.delete_collection()

        except Exception as error:
            print("Collection cleanup:", error)

    st.session_state.document_key = document_key
    st.session_state.vectorstore = None
    st.session_state.messages = []
    st.session_state.pages = 0
    st.session_state.chunks = 0


# ==========================================
# DOCUMENT METRICS
# ==========================================

with st.sidebar:

    st.divider()

    st.subheader("Active Document")
    st.text(filename)

    col1, col2 = st.columns(2)

    pages_metric = col1.empty()
    chunks_metric = col2.empty()

    pages_metric.metric(
        "Pages",
        st.session_state.pages
    )

    chunks_metric.metric(
        "Chunks",
        st.session_state.chunks
    )

    database_status = st.empty()

    if st.session_state.vectorstore is None:
        database_status.info("Ready to index")

    else:
        database_status.success("Document indexed")


# ==========================================
# WELCOME SECTION
# ==========================================

st.info(f"📄 Active document: {filename}")

welcome = st.empty()

if not st.session_state.messages:

    with welcome.container():

        st.subheader("Welcome to DocuMind AI!")

        st.write(
            "Ask questions about your uploaded PDF "
            "and inspect the retrieved source passages."
        )

        st.markdown("""
**Try asking:**

- Summarize this document.
- What are the key points?
- Explain the main concepts.
- What important information is mentioned?
""")

        st.caption(
            "Your document will be indexed automatically "
            "when you ask your first question."
        )


# ==========================================
# CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        show_sources(
            message.get("sources", [])
        )


# ==========================================
# USER INPUT
# ==========================================

question = st.chat_input(
    "Ask anything about your PDF..."
)


# ==========================================
# QUESTION PROCESSING
# ==========================================

if question and question.strip():

    welcome.empty()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        sources = []

        try:

            with st.spinner(
                "Loading local AI models..."
            ):

                embedding, chat = initialize_models()

            # Index the selected document once
            if st.session_state.vectorstore is None:

                database_status.info(
                    "Indexing document..."
                )

                with st.spinner(
                    "Processing your PDF..."
                ):

                    vectorstore, pages, chunks = process_pdf(
                        pdf_bytes,
                        filename,
                        embedding
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.pages = pages
                    st.session_state.chunks = chunks

                pages_metric.metric("Pages", pages)
                chunks_metric.metric("Chunks", chunks)

                database_status.success(
                    "Document indexed"
                )

            # Retrieve relevant passages
            with st.spinner(
                "Searching your document..."
            ):

                retrieved_docs = retrieve_documents(
                    vectorstore=st.session_state.vectorstore,
                    question=question,
                    chunk_count=st.session_state.chunks
                )

            sources = prepare_sources(retrieved_docs)

            context = format_context(sources)

            # Generate the answer
            if not context:

                response = (
                    "I cannot find this information "
                    "in the document."
                )

                st.write(response)

            else:

                chain = create_chain(chat)

                response = st.write_stream(
                    chain.stream({
                        "question": question,
                        "context": context
                    })
                )

            show_sources(sources)

        except Exception as error:

            response = f"An error occurred: {error}"

            st.error(response)
            st.exception(error)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })
