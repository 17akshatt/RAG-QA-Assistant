# The web app. Streamlit turns this plain Python script into a page you
# can open in a browser -- a text box, a button, and the results below it.
#
# This file also handles one deployment detail: where the Gemini API key
# comes from differs between your machine and Streamlit Community Cloud.
#
# Note on retrieval: rag.py's answer() function only returns the final
# answer text plus a deduped list of source filenames -- not the raw chunk
# text, which we need here to show a snippet on each source card. So this
# file does its own retrieval + prompt-building instead, reusing rag.py's
# already-built Chroma collection, TOP_K, and SYSTEM_INSTRUCTION (read-only)
# so the retrieval logic itself never has to change.

import os
import sys

import chromadb
import streamlit as st
from dotenv import load_dotenv
from google.genai import errors as genai_errors

import ingest

# Locally, this reads GEMINI_API_KEY from your .env file, same as every
# other script in this project.
load_dotenv()

# On Streamlit Community Cloud there is no .env file -- instead, you paste
# the key into the app's "Secrets" settings in the dashboard, and Streamlit
# exposes it as st.secrets. If the key wasn't already found in .env, fall
# back to st.secrets and copy it into the environment, so the rest of the
# app doesn't need to know which case it's in.
if not os.environ.get("GEMINI_API_KEY"):
    try:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass  # no .env and no secrets.toml -- the Gemini call will fail with a clear error below


def index_exists():
    """Check whether the Chroma collection ingest.py builds already exists."""
    client = chromadb.PersistentClient(path=ingest.CHROMA_DIR)
    return ingest.COLLECTION_NAME in [c.name for c in client.list_collections()]


# The ./chroma folder is listed in .gitignore (it's a generated index, not
# source code), so a fresh deployment -- like on Streamlit Community Cloud
# -- starts with no index at all. Locally you'd run `python ingest.py`
# yourself; here, build it automatically the first time the app runs so
# the deployed version doesn't crash with "no document index found".
if not index_exists():
    with st.spinner("Building document index for the first time... this only happens once."):
        ingest.build_index()

# Import AFTER the key is in the environment and the index is built:
# llm.py creates the Gemini client as soon as it's imported, and rag.py
# connects to the Chroma collection as soon as it's imported.
import llm
import rag

EXAMPLE_QUESTIONS = [
    "What is a REST API?",
    "How do I create a Python virtual environment?",
    "What does a 429 status code mean?",
    "How do I create a new Git branch?",
]

SNIPPET_LENGTH = 180

st.set_page_config(page_title="RAG Q&A Assistant", page_icon="📚")

# A little custom CSS for the answer/source "cards" -- Streamlit doesn't
# have a built-in card component, so this just styles plain <div>s.
st.markdown(
    """
    <style>
    .answer-card {
        background-color: #F3F4F6;
        border-left: 4px solid #4F46E5;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin: 0.5rem 0 1.2rem 0;
        line-height: 1.6;
        color: #1F2937;
    }
    .source-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.6rem;
    }
    .source-card-title {
        font-weight: 600;
        color: #4F46E5;
        margin-bottom: 0.2rem;
    }
    .source-card-snippet {
        color: #6B7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("How it works")
    st.write(
        "This assistant answers questions using only the documents in "
        "`docs/`. It finds the most relevant chunks by meaning (semantic "
        "search, not keyword matching), hands them to Gemini as context, "
        "and instructs it to answer strictly from that context -- citing "
        "its sources and honestly saying \"I don't know\" when the answer "
        "isn't there."
    )
    st.divider()
    st.subheader("Loaded documents")
    doc_files = sorted(
        f for f in os.listdir(ingest.DOCS_DIR)
        if os.path.isfile(os.path.join(ingest.DOCS_DIR, f))
    )
    for doc_file in doc_files:
        st.write(f"📄 {doc_file}")

st.title("📚 RAG Q&A Assistant")
st.caption("Ask a question and get an answer grounded in your own documents, with sources.")

# The text input's value lives in st.session_state under this key, so
# example-question buttons (below) can fill it in programmatically.
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def _use_example(example_text):
    """Fill the input with an example question and mark it to run."""
    st.session_state.question_input = example_text
    st.session_state.pending_question = example_text


st.text_input(
    "Your question",
    key="question_input",
    placeholder="e.g. How do I create a Python virtual environment?",
)

st.caption("Or try an example:")
example_cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, example in zip(example_cols, EXAMPLE_QUESTIONS):
    col.button(example, on_click=_use_example, args=(example,), key=f"example_{example}")

ask_clicked = st.button("Ask", type="primary")

# Figure out which question (if any) to run this script rerun: either the
# user typed one and clicked Ask, or an example button was just clicked.
question_to_run = None
if ask_clicked and st.session_state.question_input.strip():
    question_to_run = st.session_state.question_input
elif st.session_state.pending_question:
    question_to_run = st.session_state.pending_question
    st.session_state.pending_question = None
elif ask_clicked:
    st.warning("Type a question first.")

if question_to_run:
    try:
        with st.spinner("Searching documents..."):
            results = rag._collection.query(query_texts=[question_to_run], n_results=rag.TOP_K)
            chunks = results["documents"][0]
            metadatas = results["metadatas"][0]
            sources = [metadata["source"] for metadata in metadatas]

        # Same prompt format as rag.answer() -- kept identical so this
        # answer is grounded exactly the same way.
        context = "\n\n---\n\n".join(
            f"[Source: {source}]\n{chunk}" for chunk, source in zip(chunks, sources)
        )
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question_to_run}\n\n"
            "Answer the question using ONLY the context above."
        )

        with st.spinner("Thinking..."):
            answer_text = llm.ask(prompt, system_instruction=rag.SYSTEM_INSTRUCTION, temperature=0.2)

        st.subheader("Answer")
        st.markdown(f'<div class="answer-card">{answer_text}</div>', unsafe_allow_html=True)

        st.subheader("Sources")
        for chunk_text, source in zip(chunks, sources):
            snippet = chunk_text[:SNIPPET_LENGTH].strip()
            if len(chunk_text) > SNIPPET_LENGTH:
                snippet += "…"
            st.markdown(
                f'''
                <div class="source-card">
                    <div class="source-card-title">📄 {source}</div>
                    <div class="source-card-snippet">{snippet}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    except genai_errors.APIError as e:
        # Log the real error server-side (visible in Streamlit Cloud's
        # app logs) so it can be debugged, but never show a raw
        # traceback to the user -- just a plain-English message.
        print(f"Gemini API error: {e}", file=sys.stderr)
        if e.code == 429:
            st.error("The daily API quota has been reached — please try again tomorrow.")
        else:
            st.error("Something went wrong talking to the AI model. Please try again in a moment.")
