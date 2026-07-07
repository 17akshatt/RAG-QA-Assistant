# The web app. Streamlit turns this plain Python script into a page you
# can open in a browser -- a text box, a button, and the results below it.
#
# This file also handles one deployment detail: where the Gemini API key
# comes from differs between your machine and Streamlit Community Cloud.

import os

import streamlit as st
from dotenv import load_dotenv

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
        pass  # no .env and no secrets.toml -- ask() will fail with a clear error below

# Import AFTER the key is in the environment: llm.py creates the Gemini
# client as soon as it's imported, so the key needs to be set up first.
from rag import answer

st.set_page_config(page_title="RAG Q&A Assistant", page_icon="📚")

st.title("📚 RAG Q&A Assistant")
st.write(
    "Ask a question about the documents in `docs/`. The assistant answers "
    "using only those documents, and shows its sources below the answer."
)

question = st.text_input("Your question", placeholder="e.g. How do I create a Python virtual environment?")
ask_clicked = st.button("Ask")

if ask_clicked and question.strip():
    with st.spinner("Thinking..."):
        answer_text, sources = answer(question)

    st.subheader("Answer")
    st.write(answer_text)

    if sources:
        with st.expander("Sources"):
            for source in sources:
                st.write(f"- {source}")
elif ask_clicked:
    st.warning("Type a question first.")
